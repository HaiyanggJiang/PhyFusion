# config.py
import torch
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
