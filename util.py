# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from torch.nn import functional as F
from math import exp
import math
from device import device
import numpy as np

def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res

def rot(img, rot_mode):
    if rot_mode == 0:  # 90 degrees clockwise
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:  # 180 degrees
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:  # 270 degrees clockwise (or 90 degrees counterclockwise)
        img = img.transpose(-2, -1)
        img = img.flip(-1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)  # Vertical flip
    elif flip_mode == 1:
        img = img.flip(-1)  # Horizontal flip
    return img
def randrot(img):
    mode = np.random.randint(0, 3)  # Rotating in 90-degree increments
    return rot(img, mode)

def randfilp(img):
    mode = np.random.randint(0, 2)  # Flipping either vertically or horizontally
    return flip(img, mode)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   


def std1(img, window_size=9):
    
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)  
    mu = F.conv2d(img, window, padding=padd, groups=channel)  
    mu_sq = mu.pow(2)
    sigma_sq = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq  
    sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))  
    return sigma


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

class TeVloss():
    def __init__(self, vnums=4, loss_type='MSE'):
        self.vnums = vnums
        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        else:
            pass
    
    def loss_rec(self, preds, x):
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        rec_img = self.rec(preds, x)
        loss = self.loss(rec_img, x_mean)
        return loss
    
    def rec(self, preds, x):
        x_mean = torch.mean(x, dim=1)
        e = self.rec_e(preds).to(device)
        T = self.rec_T(preds).to(device)
        env = self.rec_env(preds, x_mean)
        rec_img = torch.mul(e, T) + torch.mul(1-e, env)
        return rec_img
    
    def rec_e(self, preds):
        e = preds[:,0,:,:].unsqueeze(1)
        return e
    
    def rec_T(self, preds):
        T = preds[:,1,:,:].unsqueeze(1)
        return T
    
    def rec_env(self, preds, x_mean):
        b, _, h, w = preds.shape
        V = preds[:, 2:2+self.vnums, :, :]
        h_split_nums = int(math.sqrt(self.vnums))
        w_split_nums = self.vnums // h_split_nums
        assert h_split_nums * w_split_nums == self.vnums
        x_beta = F.avg_pool2d(x_mean, (h // h_split_nums, w // w_split_nums)).reshape(b, 1, self.vnums)
        v_pred = V.reshape(b, self.vnums, h*w).to(device)
        env = torch.matmul(x_beta, v_pred)
        env = env.view(b, 1, h, w)
        return env

class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = GradientMaxLoss()
        self.loss_func_color = L_color()
    def forward(self, image_visible, image_infrared, image_fused):
        loss_ssim = self.loss_func_ssim(image_visible, image_fused) + self.loss_func_ssim(image_infrared, image_fused)
        loss_Grad = self.loss_func_Grad(image_visible, image_infrared, image_fused)
        loss_int = torch.mean(torch.pow((image_fused - image_visible), 2)) + torch.mean((image_fused < image_infrared) * torch.abs((image_fused - image_infrared)))

        return loss_ssim, loss_Grad, loss_int
    
    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.to(device)
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret

# use the GradientMaxLoss or L_Grad
class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).to(device)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).to(device)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)


   
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image