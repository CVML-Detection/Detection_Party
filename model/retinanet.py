import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
from option import device


class RetinaNet(nn.Module):
    def __init__(self, base, n_classes=21, C3_size=512, C4_size=1024, C5_size=2048, feature_size=256):
        super(RetinaNet, self).__init__()
        self.base = base
        self.n_classes = n_classes

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        #self.cls_final_module = nn.Conv2d(256, 9 * 21, kernel_size=3, padding=1)
        self.cls_module = CLS_Module(num_class = self.n_classes)
        self.reg_module = REG_Module()

        # fpn init
        self.init_fpn()
        self.init_subsets()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_fpn(self):
        self.init_layer(self.P5_1)
        self.init_layer(self.P5_2)
        self.init_layer(self.P4_1)
        self.init_layer(self.P4_2)
        self.init_layer(self.P3_1)
        self.init_layer(self.P3_2)
        self.init_layer(self.P6)
        self.init_layer(self.P7_2)

    def init_layer(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

    def init_subsets(self):
        i = 0
        for c in self.cls_module.features.children():

            if isinstance(c, nn.Conv2d):
                if i == 8:  # final layer

                    pi = 0.01
                    b = - math.log((1 - pi) / pi)
                    nn.init.constant_(c.bias, b)
                    nn.init.normal_(c.weight, std=0.01)

                else:
                    # nn.init.xavier_uniform_(c.weight)
                    # nn.init.uniform_(c.bias, 0.)

                    nn.init.normal_(c.weight, std=0.01)
                    nn.init.constant_(c.bias, 0)

            i += 1

        for c in self.reg_module.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.uniform_(c.bias, 0.)

    def _upsample(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.interpolate(x, [h, w], mode='nearest')

        return x_upsampled

    def forward(self, inputs):
        C3, C4, C5 = self.base(inputs)

        # features = []
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self._upsample(P5_x, C4)
        P5_x = self.P5_2(P5_x)
        # print(P5_x.size())         # torch.Size([4, 256, 19, 19])

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self._upsample(P4_x, C3)
        P4_x = self.P4_2(P4_x)
        # print(P4_x.size())         # torch.Size([4, 256, 38, 38])

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        features = [P3_x, P4_x, P5_x, P6_x, P7_x]

        reg = torch.cat([self.reg_module(feature) for feature in features], dim=1)
        cls = torch.cat([self.cls_module(feature) for feature in features], dim=1)

        return reg, cls


class CLS_Module(nn.Module):
    def __init__(self, num_class):
        super(CLS_Module, self).__init__()
        self.num_class = num_class
        self.features = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 9 * self.num_class, kernel_size=3, padding=1),
                                      # nn.Sigmoid()
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, self.num_class)

        return x


class REG_Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 4 * 9, kernel_size=3, padding=1),
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()   # convert B x C x W x H to B x W x H x C
        x = x.view(batch_size, -1, 4)

        return x


if __name__ == "__main__":

    from model.resnet import ResNet_50
    model = RetinaNet(base=ResNet_50(pretrained=True), n_classes=81)

    image = torch.randn([2, 3, 800, 800 ])
    a,b=model(image)
    print(a.shape)
    print(b.shape)