# coding: utf-8
import torch
import argparse
import os
import numpy as np
from pathlib import Path
import cv2
import torch.nn as nn
from device import device
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from dino.dino_net import DinoFeaturizer
from model.net import Restormer_Encoder, Restormer_Decoder
from model.tevmodel import TeVNet
from model.DeCOM import DecomNet
from tqdm import tqdm
from PIL import Image

from collections import OrderedDict
from util import (
    RGB2YCrCb,
    YCbCr2RGB)

def save_img_single(img, name, width=None, height=None):
    img = tensor2img(img)               # [C,H,W] tensor -> uint8 HWC
    img = Image.fromarray(img)
    if width is not None and height is not None:
        w = int(width.numpy()) if hasattr(width, "numpy") else int(width)
        h = int(height.numpy()) if hasattr(height, "numpy") else int(height)
        img = img.resize((w, h))
    img.save(name)

def tensor2img(img):
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    mmin, mmax = img.min(), img.max()
    if mmax > mmin:
        img = (img - mmin) / (mmax - mmin)
    else:
        img = np.zeros_like(img)
    img = np.transpose(img, (1, 2, 0)) * 255.0
    return img.astype(np.uint8)

def test_all(path=os.path.join(os.getcwd(), 'OUTPUT', 'LLVIP_500'), data='assets/data', DIDF_Encoder=None, DIDF_Decoder=None, args = None):
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = Fusion_dataset('val', ir_path=os.path.join(data, 'LLVIP_500_Train', 'ir'),vi_path=os.path.join(data, 'LLVIP_500_Train', 'vis'))
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 8,
        pin_memory=True,
        drop_last=False)
    
    test_loader.n_iter = len(test_loader)
    
    if DIDF_Encoder is None:
        DIDF_Encoder = Restormer_Encoder().to(device)
        DIDF_Decoder = Restormer_Decoder().to(device)
        ckpt_path = "CHECKPOINT/best_model.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu")

        safe_load_into(DIDF_Encoder, ckpt["DIDF_Encoder"], name="DIDF_Encoder")
        safe_load_into(DIDF_Decoder, ckpt["DIDF_Decoder"], name="DIDF_Decoder")
        DIDF_Encoder.eval()
        DIDF_Decoder.eval()
        net = DinoFeaturizer().to(device)
        TeV = TeVNet(in_channels=3, out_channels=6, args=args).to(device)
        Decom =DecomNet(layer_num=5, channel=64, kernel_size=3,args = args).to(device)

        # Load model weights
        state_dict = TeV.state_dict()
        ckpt = torch.load(args.weights_file, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)


        model_has_module = any(k.startswith("module.") for k in state_dict.keys())
        ckpt_has_module  = any(k.startswith("module.") for k in sd.keys())
        if model_has_module and not ckpt_has_module:

            sd = {f"module.{k}": v for k, v in sd.items()}
        elif (not model_has_module) and ckpt_has_module:

            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}


        loaded, skipped = 0, []
        for n, p in sd.items():
            if n in state_dict:
                if state_dict[n].shape == p.shape:
                    state_dict[n].copy_(p)
                    loaded += 1
                else:
                    skipped.append((n, tuple(state_dict[n].shape), tuple(p.shape)))

        TeV.load_state_dict(state_dict)

        print(f"[TevModel] loaded params: {loaded}, skipped: {len(skipped)}")
        if skipped:
            print("  shape-mismatch keys (first 8):",
                [f"{k}: model{ms} <- ckpt{cs}" for k, ms, cs in skipped[:8]])

        TeV.eval()

        # Load DeComNet model weights
        ckpt_decom = torch.load(args.ckpt_path, map_location="cpu")
        sd_decom = (
            ckpt_decom.get("state_dict", None)
            or ckpt_decom.get("DeComNet", None)
            or ckpt_decom.get("decomnet", None)
            or ckpt_decom
        )


        model_has_module = any(k.startswith("module.") for k in Decom.state_dict().keys())
        ckpt_has_module  = any(k.startswith("module.") for k in sd_decom.keys())
        if model_has_module and not ckpt_has_module:
            sd_decom = {f"module.{k}": v for k, v in sd_decom.items()}
        elif (not model_has_module) and ckpt_has_module:
            sd_decom = {k.replace("module.", "", 1): v for k, v in sd_decom.items()}


        dst = Decom.state_dict()
        loaded, skipped = 0, []
        if any(k.startswith("decom.") for k in sd_decom.keys()):
            sd_decom = {k.replace("decom.", "", 1): v for k, v in sd_decom.items()}
        for k, v in sd_decom.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k].copy_(v)
                loaded += 1
            else:
                if k in dst:
                    skipped.append((k, tuple(dst[k].shape), tuple(v.shape)))
            Decom.load_state_dict(dst)
        missing, unexpected = Decom.load_state_dict(dst, strict=False)

        print(f"[DeComNet] loaded params: {loaded}, skipped: {len(skipped)}")
        if skipped:
            print('  shape-mismatch keys (first 8):',
                [f"{k}: model{ms} <- ckpt{cs}" for k, ms, cs in skipped[:8]])
        if missing:
            print("  missing keys (first 8):", list(missing)[:8])
        if unexpected:
            print("  unexpected keys (first 8):", list(unexpected)[:8])

        Decom.eval()


    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            widths, heights = None, None
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            tev_pred = TeV(img_ir)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            R,L = Decom(img_vis)
            feature, feature_dino, *_ = DIDF_Encoder(img_vis, img_ir, R, tev_pred, net)
            fused, _ = DIDF_Decoder(feature, feature_dino)
            fused_rgb = YCbCr2RGB(fused, vi_Cb, vi_Cr)
            B = fused_rgb.size(0)
            for i in range(B):
                img_name = str(name[i])
                fusion_save_name = os.path.join(str(out_dir), img_name)

                w_i = widths[i] if widths is not None else None
                h_i = heights[i] if heights is not None else None

                save_img_single(fused_rgb[i, ...], fusion_save_name, w_i, h_i)

def save_pic(outputpic, save_path):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    cv2.imwrite(save_path, outputpic)

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def safe_load_into(model, src_sd, name="model"):

    dst_sd = model.state_dict()

    model_has_module = any(k.startswith("module.") for k in dst_sd.keys())
    ckpt_has_module  = any(k.startswith("module.") for k in src_sd.keys())
    if model_has_module and not ckpt_has_module:
        src_sd = {f"module.{k}": v for k, v in src_sd.items()}
    elif (not model_has_module) and ckpt_has_module:
        src_sd = {k.replace("module.", "", 1): v for k, v in src_sd.items()}

    loaded, skipped = 0, []
    for k, v in src_sd.items():
        if k in dst_sd:
            if dst_sd[k].shape == v.shape:
                dst_sd[k].copy_(v)
                loaded += 1
            else:
                skipped.append((k, tuple(dst_sd[k].shape), tuple(v.shape)))

    model.load_state_dict(dst_sd)
    print(f"[resume] {name}: loaded={loaded}, skipped={len(skipped)}")
    if skipped:
        print("  shape-mismatch keys (first 8):")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training Example')
    parser.add_argument('--smp_model', type=str, default='DeepLabV3Plus')
    parser.add_argument('--smp_encoder', type=str, default='efficientnet-b4')
    parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')
    parser.add_argument('--weights_file',type=str, default="CHECKPOINT/IR.pth", help="")
    parser.add_argument('--ckpt_path',type=str, default="CHECKPOINT/VIS.pth", help="")
    args = parser.parse_args()
    
    test_all(args = args)
