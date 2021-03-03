import torch

device_ids = [2, 3]
device_ids = [1]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
