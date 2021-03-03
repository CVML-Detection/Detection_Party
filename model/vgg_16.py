
import torch.nn as nn
import torch
import math
import torchvision


class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv1

                                      nn.Conv2d(64, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv2

                                      nn.Conv2d(128, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                      # conv3

                                      nn.Conv2d(256, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv4

                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                      # conv5

                                      nn.Conv2d(512, 1024, 3, padding=6, dilation=6),
                                      nn.ReLU(inplace=True),
                                      # conv6

                                      nn.Conv2d(1024, 1024, 1, padding=0),
                                      nn.ReLU(inplace=True),
                                      # conv7
                                      )
        self.init_conv2d()

        if pretrained:

            std = torchvision.models.vgg16(pretrained=True).features.state_dict()
            model_dict = self.features.state_dict()
            pretrained_dict = {k: v for k, v in std.items() if k in model_dict}  # 여기서 orderdict 가 아니기 때문에
            model_dict.update(pretrained_dict)
            self.features.load_state_dict(model_dict)

    def init_conv2d(self):
        for c in self.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.features(x)
        return x

