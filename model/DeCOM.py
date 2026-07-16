import torch
import torch.nn as nn
import torch.nn.functional as F


def concat(layers):
    return torch.cat(layers, dim=1)


class DecomNet(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3,args = None):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num

        self.shallow_conv = nn.Conv2d(4, channel, kernel_size * 3, padding=(kernel_size * 3)//2, bias=True)
        self.hidden_convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size, padding=kernel_size//2, bias=True)
            for _ in range(layer_num)
        ])
        self.recon_conv = nn.Conv2d(channel, 4, kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x):
        input_max, _ = torch.max(x, dim=1, keepdim=True)   
        input_im = concat([input_max, x])                  

        conv = self.shallow_conv(input_im)
        for conv_layer in self.hidden_convs:
            conv = F.relu(conv_layer(conv))
        conv = self.recon_conv(conv)

        R = torch.sigmoid(conv[:, 0:3, :, :])
        L = torch.sigmoid(conv[:, 3:4, :, :])
        return R, L
