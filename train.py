# coding: utf-8
import os
import sys
import torch
from time import time
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from dino.dino_net import DinoFeaturizer
from model.net import Restormer_Encoder, Restormer_Decoder
import torch.optim as optim
import itertools
from tqdm import tqdm
from util import (
    fusion_loss,
    TeVloss,
    create_lr_scheduler,
    RGB2YCrCb,
    YCbCr2RGB)
import torch.nn as nn
from model.tevmodel import TeVNet
from model.DeCOM import DecomNet
from device import device
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

class TeVTrain:
    def __init__(self, args):
        self.args = args
        self.DIDF_Encoder = Restormer_Encoder().to(device) 
        self.DIDF_Decoder = Restormer_Decoder().to(device)
        self.net = DinoFeaturizer().to(device)

        self.net.eval()

        self.TeVNet = TeVNet(in_channels=3, out_channels=6, args=self.args).to(device)
        self.DeComNet = DecomNet(layer_num=5, channel=64, kernel_size=3,args =self.args).to(device)
        

        # Load TeVNet model weights
        state_dict = self.TeVNet.state_dict()
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
        self.TeVNet.load_state_dict(state_dict)

        print(f"[TevModel] loaded params: {loaded}, skipped: {len(skipped)}")
        if skipped:
            print("  shape-mismatch keys (first 8):",
                [f"{k}: model{ms} <- ckpt{cs}" for k, ms, cs in skipped[:8]])
        
        self.TeVNet.eval()

        # Load DeComNet model weights
        ckpt_decom = torch.load(args.ckpt_path, map_location="cpu")
        sd_decom = (
            ckpt_decom.get("state_dict", None)
            or ckpt_decom.get("DeComNet", None)
            or ckpt_decom.get("decomnet", None)
            or ckpt_decom
        )

      
        model_has_module = any(k.startswith("module.") for k in self.DeComNet.state_dict().keys())
        ckpt_has_module  = any(k.startswith("module.") for k in sd_decom.keys())
        if model_has_module and not ckpt_has_module:
            sd_decom = {f"module.{k}": v for k, v in sd_decom.items()}
        elif (not model_has_module) and ckpt_has_module:
            sd_decom = {k.replace("module.", "", 1): v for k, v in sd_decom.items()}

        
        dst = self.DeComNet.state_dict()
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
            self.DeComNet.load_state_dict(dst)
        missing, unexpected = self.DeComNet.load_state_dict(dst, strict=False)

        print(f"[DeComNet] loaded params: {loaded}, skipped: {len(skipped)}")
        if skipped:
            print('  shape-mismatch keys (first 8):',
                [f"{k}: model{ms} <- ckpt{cs}" for k, ms, cs in skipped[:8]])
        if missing:
            print("  missing keys (first 8):", list(missing)[:8])
        if unexpected:
            print("  unexpected keys (first 8):", list(unexpected)[:8])

        self.DeComNet.eval()

        self.TeVloss = TeVloss()
        self.Fusionloss = fusion_loss()
        cudnn.benchmark = True

    def train_step(self, optimizer, lr_scheduler, image_vi, image_ir):
        self.DIDF_Encoder.train()
        self.DIDF_Decoder.train()
        l_g_total = 0
        optimizer.zero_grad()

        tev_pred = self.TeVNet(image_ir)
        R, L = self.DeComNet(image_vi)

        feature,feature_dino, feature_c1, feature_s1, feature_c2, feature_s2, mask_ir = self.DIDF_Encoder(image_vi, image_ir, R, tev_pred, self.net)
        data_Fuse, feature_F = self.DIDF_Decoder(feature, feature_dino)
        
        fusion_tev = data_Fuse.expand(-1, 3, -1, -1)
        rec_pred = self.TeVNet(fusion_tev)
       

        tev_pred1 = self.TeVNet(image_ir)
        loss_tev = torch.nn.functional.mse_loss(tev_pred1, rec_pred, reduction='mean').to(device)

        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(image_vi)
        loss_ssim, loss_Grad, loss_int = self.Fusionloss(Y_vi, image_ir[:, :1, ...], data_Fuse)
        
        l_g_total =loss_tev + loss_ssim + 10 * loss_Grad + loss_int

        l_g_total.backward()
        lr_fusion = optimizer.param_groups[0]["lr"]
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        return l_g_total, loss_ssim, loss_Grad, loss_int, lr_fusion
    
    def train(self):
        train_dataset = Fusion_dataset('train')
        print("the training dataset is length:{}".format(len(train_dataset)))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True)
        ep_iter = len(train_loader)
        max_iter = self.args.epoch * ep_iter
        print("Training...iter: {}".format(max_iter))
        optimizer = optim.AdamW(
            [
                {"params": [p for p in self.DIDF_Encoder.parameters() if p.requires_grad], "lr": 1e-4, "name": "enc"},
                {"params": [p for p in self.DIDF_Decoder.parameters() if p.requires_grad], "lr": 1e-4, "name": "dec"},
            ],
            weight_decay=5e-2
        )
        lr_scheduler = create_lr_scheduler(optimizer, ep_iter, self.args.epoch, warmup=True)

        current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
        summary_dir = os.path.join(self.args.summary_dir, 'Train_Fusion' + current_time)
        with SummaryWriter(summary_dir) as writer:    
            min_l_g_total = float('inf')  
            Allepochs = self.args.epoch
            global_step = 0
            start_epoch = 1
            checkpoint_dir = self.args.savePTH
            if os.path.exists(checkpoint_dir):
                files = os.listdir(checkpoint_dir)
                epochs = [int(f[5:]) for f in files if f.startswith('epoch') and f[5:].isdigit()]
                if epochs:
                    model_save_path = os.path.join(checkpoint_dir, 'epoch{}'.format(max(epochs)), 'best_model.pth')
                    checkpoint = torch.load(model_save_path, map_location='cpu')
                    
                    
                    if all(k in checkpoint for k in ["DIDF_Encoder", "DIDF_Decoder", "net"]):
                        self._safe_load_into(self.DIDF_Encoder, checkpoint["DIDF_Encoder"], name="DIDF_Encoder")
                        self._safe_load_into(self.DIDF_Decoder, checkpoint["DIDF_Decoder"], name="DIDF_Decoder")
                    
                    elif "DIDF_Encoder" in checkpoint:
                        print("[resume] 检测到旧 ckpt（包含 'DIDF_Encoder'）。")
                    else:
                        print("[resume] 未找到可识别的模型权重键，跳过模型恢复。")

                    
                    if "optimizer" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if torch.is_tensor(v):
                                    state[k] = v.to(device)

                    if "lr_scheduler" in checkpoint:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

                    start_epoch   = checkpoint.get("ep", 0) + 1
                    global_step   = checkpoint.get("total_it", 0)
                    min_l_g_total = checkpoint.get("min_l_g_total", min_l_g_total)

                    
                    lr = optimizer.param_groups[0]['lr']
                    print(f'lr_fusion= {lr:.10f}')
                    print('Resuming training from epoch {}'.format(start_epoch))
            
            start = glob_st = time()
            for epoch in range(start_epoch, Allepochs + 1):
                data_loader = tqdm(train_loader, file=sys.stdout)
                epoch_loss_sum = 0.0
                epoch_iter_cnt = 0

                for it, (image_vis, image_ir, names) in enumerate(data_loader):
                    global_step += 1
                    batch_images_vi = image_vis.to(device)
                    batch_images_ir = image_ir.to(device)

                    total_loss, loss_ssim, loss_Grad, loss_int, lr_fusion = \
                        self.train_step(optimizer, lr_scheduler, batch_images_vi, batch_images_ir)


                    tl = float(total_loss.item() if hasattr(total_loss, "item") else total_loss)
                    epoch_loss_sum += tl
                    epoch_iter_cnt += 1

                    data_loader.desc = (
                        "[train epoch {}] loss: {:.3f} loss_ssim: {:.3f}  grad loss: {:.3f} loss_int: {:.3f} "
                        "lr: {:.6f}"
                    ).format(epoch, total_loss, loss_ssim, loss_Grad, loss_int, lr_fusion)


                end = time()
                training_time, glob_t_intv = end - start, end - glob_st
                eta = int((Allepochs * len(train_loader) - global_step) * (glob_t_intv / max(global_step, 1)))
                eta = str(timedelta(seconds=eta))
                print(f'Still need {eta}')
                start = time()

                
                epoch_loss_avg = epoch_loss_sum / max(epoch_iter_cnt, 1)

                
                if epoch_loss_avg < 0:
                    print(f'[epoch {epoch}] Invalid loss, skipping update and model save.')
                    improved = False
                else:
                    improved = epoch_loss_avg < min_l_g_total
                    if improved:
                        min_l_g_total = epoch_loss_avg

                print(f'[epoch {epoch}] avg_total_loss={epoch_loss_avg:.6f} | best(min_l_g_total)={min_l_g_total:.6f} | improved={improved}')


                if epoch > 90:
                    if improved or epoch == Allepochs:
                        print('Saving model (improved min_l_g_total or last epoch after 450)...')
                        epoch_folder = os.path.join(self.args.savePTH, f'epoch{epoch}')
                        os.makedirs(epoch_folder, exist_ok=True)
                        save_file = {
                            # "model": self.fusionNet.state_dict(),
                            "DIDF_Encoder": self.DIDF_Encoder.state_dict(),
                            "DIDF_Decoder": self.DIDF_Decoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "ep": epoch,
                            "total_it": global_step,
                            "min_l_g_total": min_l_g_total
                        }
                        torch.save(save_file, os.path.join(epoch_folder, "best_model.pth"))
                        print(f'Epoch {epoch}/{Allepochs}, Model saved at {epoch_folder}, min_l_g_total: {min_l_g_total:.6f}')
                    else:
                        print(f'Model not saved (epoch>{90}). epoch_avg_loss: {epoch_loss_avg:.6f}, best: {min_l_g_total:.6f}')
                else:
                    print(f'Skip saving before or at epoch 90. (epoch={epoch})')
                    if epoch == Allepochs and Allepochs <= 90:
                        print('Final epoch <= 450, saving anyway...')
                        epoch_folder = os.path.join(self.args.savePTH, f'epoch{epoch}')
                        os.makedirs(epoch_folder, exist_ok=True)
                        save_file = {
                            # "model": self.fusionNet.state_dict(),
                            "DIDF_Encoder": self.DIDF_Encoder.state_dict(),
                            "DIDF_Decoder": self.DIDF_Decoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "ep": epoch,
                            "total_it": global_step,
                            "min_l_g_total": min_l_g_total
                        }
                        torch.save(save_file, os.path.join(epoch_folder, "best_model.pth"))
                        print(f'Epoch {epoch}/{Allepochs}, Model saved at {epoch_folder}, min_l_g_total: {min_l_g_total:.6f}')

                       
    def _safe_load_into(self, model, src_sd, name="model"):
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
            print("  shape-mismatch keys (first 8):",
                [f"{k}: model{ms} <- ckpt{cs}" for k, ms, cs in skipped[:8]])


