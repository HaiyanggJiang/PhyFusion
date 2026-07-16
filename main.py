# coding: utf-8
import argparse
import os
from train import TeVTrain
import pprint

parser = argparse.ArgumentParser(description='PyTorch Training Example')

parser.add_argument('--epoch', type=int,default=120,help="Number of epoch [10]")
parser.add_argument('--batch_size',type=int,default=4, help="The size of batch images [128]")
parser.add_argument('--checkpoint',type=str, default="CHECKPOINT", help="Name of checkpoint directory [checkpoint]")
parser.add_argument('--savePTH',type=str, default="savePTH", help="savePTH")
parser.add_argument('--summary_dir',type=str, default="log", help="Name of log directory [log]")
parser.add_argument('--weights_file',type=str, default="/CHECKPOINT/tev_msrs.pth", help="")
parser.add_argument('--ckpt_path',type=str, default="/CHECKPOINT/DeCOM.pth", help="")
parser.add_argument('--smp_model', type=str, default='DeepLabV3Plus')
parser.add_argument('--smp_encoder', type=str, default='efficientnet-b4')
parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')

args = parser.parse_args()
pp = pprint.PrettyPrinter()

def main(args):
    pp.pprint(vars(args))

    os.makedirs(args.checkpoint, exist_ok=True)
    
    model = TeVTrain(args)
    model.train()

if __name__ == '__main__':
    main(args)