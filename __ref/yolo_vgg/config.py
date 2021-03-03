import torch
from option import opts

if opts.paeng:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
