import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import matplotlib.pyplot as plt

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x




import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,stride=1, padding=1, bias=bias)
        #nn.Conv2d(in_c, embed_dim, kernel_size=8, stride=1)

    def forward(self, x):
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, shared_dim=256, n_blk=3):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, output_dim*2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        batch = x.shape[0]
        weight, bias = self.MLP(x).view(batch, -1, 1, 1).chunk(2, dim=1)

        return weight, bias
class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self,num_features=96,eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.gamma = None
        self.beta = None
        self.eps = eps
    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in * self.gamma + self.beta
        return out

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class AdaptiveBatchNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.param_free_norm = nn.BatchNorm2d(num_features, affine=False)
        # self.param_free_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = None
        self.beta = None

    def forward(self, x):
        normalized = self.param_free_norm(x)
        out = normalized * (1 + self.gamma) + self.beta
        return out
class ResidualBlock(nn.Module):
    def __init__(self, features, norm):
        super(ResidualBlock, self).__init__()
        if norm == 'adain':

            norm_layer = AdaptiveInstanceNorm2d
        elif norm == "spade":  
            norm_layer = AdaptiveBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )


    def forward(self, x):
       return x + self.block(x)
class CrossAttention(nn.Module):
    def __init__(self, dim, down):
        super().__init__()
        self.heads = 4
        self.scale = (dim // 4) ** -0.5

        self.to_q = nn.Linear(dim//down, dim//down, bias=False)
        self.to_k = nn.Linear(dim, dim//down, bias=False)
        self.to_v = nn.Linear(dim, dim//down, bias=False)
        #self.to_out = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input):
        B, C, H, W = q_input.shape
        q = q_input.flatten(2).permute(0, 2, 1)  # [B, HW_q, C]
        kv = kv_input.flatten(2).permute(0, 2, 1) # [B, HW_kv, C]

        q = self.to_q(q)
        k = self.to_k(kv).permute(0, 2, 1)
        v = self.to_v(kv)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, HW_q, C]
        out = q_input+out.permute(0, 2, 1).view(B, C, H, W)

        return out


class SpatialChannelGatedFusion(nn.Module):
    def __init__(self, dim, dino_dim,mid_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3*dim, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim, kernel_size=1)
        )
        self.up = nn.Sequential(nn.Upsample(scale_factor=8),
                                nn.Conv2d(dino_dim // 2, dino_dim // 2, 3, padding=1),
                                nn.BatchNorm2d(dino_dim // 2),
                                nn.PReLU())

        self.c_att_S = CrossAttention(768*2, 24)
        self.c_att_M = CrossAttention(768*2, 24)
        self.c_att_H = CrossAttention(768*2, 24)
    def forward(self, out_enc_level1, feature_C1, feature_C2):
        
        fS=self.c_att_S(out_enc_level1, torch.cat([feature_C1[0], feature_C2[0]], 1))
        
        fM = self.c_att_M(out_enc_level1, torch.cat([feature_C1[1], feature_C2[1]], 1))
       
        fH = self.c_att_H(out_enc_level1, torch.cat([feature_C1[2], feature_C2[2]], 1))
        concat = torch.cat([fS, fM, fH], dim=1)  
        logits = self.conv(concat)               

        return logits
    
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=1,
                 dim=64,
                 dino_dim=768,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.cluster1 = torch.nn.Conv2d(dim * 2, dim, (1, 1))

        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.conv_dino_1 = nn.Sequential(ResidualBlock(dim, norm="adain"),
                                         ResidualBlock(dim, norm="adain"),


                                       )

        self.mlp_dino_1 = MLP(dino_dim * 2, dim)

        self.v_fuse = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.v_fuse.weight.fill_(1.0 / 4.0)
        self.phyPromt_fuse=SpatialChannelGatedFusion(dim, dino_dim)

    def assign_adain_params(self, gamma, beta):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d", "AdaptiveBatchNorm2d"]:
                m.gamma = gamma
                m.beta = beta
    def forward(self, data_VIS, data_IR, R, tev_pred, net):
        b, c, h, w = data_IR.shape
        tev_feat = tev_pred[:, 0:1, :, :].repeat(1, 3, 1, 1) # (B, e*3, H, W)
        p_feat = R

        with torch.no_grad():
            feature_C1, mask_2, feature_S1 = net(tev_feat)
            feature_C2, mask_2,feature_S2 = net(p_feat)
        inp_enc_level1_A = self.patch_embed( torch.cat([data_VIS, data_IR[:, :1, ...]], 1))
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)
        feature_C = self.phyPromt_fuse(out_enc_level1_A, feature_C1, feature_C2)
        gamma_d, beta_d = self.mlp_dino_1(torch.cat([feature_S1, feature_S2], 1))
        self.assign_adain_params(gamma_d, beta_d)

        feature_dino = self.conv_dino_1(feature_C)

        return out_enc_level1_A,feature_dino,feature_C1,feature_S1,feature_C2,feature_S2,mask_2

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()

        self.reduce_channel = nn.Conv2d(dim*2, int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)

        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, feature):

        out_enc_level1 = self.encoder_level2(feature)
        if inp_img is not None:
           
            out_enc_level1 = self.output(out_enc_level1) #+ inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
            
        return self.sigmoid(out_enc_level1), feature
    
if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()

