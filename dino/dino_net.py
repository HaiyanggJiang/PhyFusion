import torch.nn as nn
import dino.vision_transformer as vits
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
class DinoFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()

        patch_size = 16
        self.patch_size = 16
        self.feat_type = "feat"
        arch = "vit_base"
        pretrained_weights = 0
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)



        self.model.eval()
        self.dropout = torch.nn.Dropout2d(p=.1)


        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        # with torch.no_grad():
        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)

        # get selected layer activations
        feats=[]
        attns=[]
        clss=[]
        f, at, qkvs = self.model.get_intermediate_feat(img, n=n)
        for i in range(0,3):
            feat, attn, qkv = f[i], at[i], qkvs[i]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            nh = attn.shape[1]

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            cls = feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)
            attn = attn[:, :, 0, 1:].reshape(feat.shape[0], nh, feat_h, feat_w)
            image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], nh, feat_h, feat_w, -1)
            B, H, I, J, D = image_k.shape
            image_k = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)


            attn_mean = attn.mean(dim=1, keepdim=True)
            attn_max, _ = attn.max(dim=1, keepdim=True)
            attn1 = ((attn_max - attn_max.min()) / (attn_max.max() - attn_max.min() + 1e-8)) ** 0.5
            at1 = attn1

            at1 = F.interpolate(attn1, size=( img.shape[2],  img.shape[3]), mode="bilinear", align_corners=False)


            threshold1 = torch.quantile(at1, 0.8) 

            mask_attn1 = (at1 >= threshold1).float()

            feats.append(image_feat),attns.append(attn1),clss.append(cls)


        return feats , mask_attn1,clss[2]#,

class DinoResNetFeaturizer(nn.Module):
    def __init__(self, arch="dino_resnet50", device="cuda"):
        super().__init__()

       
        self.backbone = torch.hub.load('facebookresearch/dino:main', arch)
        self.backbone.eval().to(device)


        return_nodes = {
            "layer1": "feat_s4",
            "layer2": "feat_s8",
            "layer3": "feat_s16",
            "layer4": "feat_s32"
        }

        self.feat_extractor = create_feature_extractor(self.backbone, return_nodes=return_nodes).to(device)


        for p in self.feat_extractor.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats = self.feat_extractor(x)
        print(feats["feat_s8"].shape)
        return feats