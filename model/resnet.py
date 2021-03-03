from torchvision.models import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
import torch.nn as nn
import torch


class ResNet_50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet_50 = resnet50(pretrained=pretrained)
        self.resnet_50_list = nn.ModuleList(list(self.resnet_50.children())[:-2])  # to layer 1
        self.res_50 = nn.Sequential(*self.resnet_50_list)

    def forward(self, x):

        x = self.resnet_50_list[0](x)  # 7 x 7 conv 64
        x = self.resnet_50_list[1](x)  # bn
        x = self.resnet_50_list[2](x)  # relu
        x = self.resnet_50_list[3](x)  # 3 x 3 maxpool

        x = self.resnet_50_list[4](x)  # layer 1
        c3 = x = self.resnet_50_list[5](x)  # layer 2
        c4 = x = self.resnet_50_list[6](x)  # layer 3
        c5 = x = self.resnet_50_list[7](x)  # layer 4

        # print(x.size())          # torch.Size([4, 2048, 19, 19])
        return [c3, c4, c5]


class ResNet_101(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet_101 = resnet101(pretrained=pretrained)
        self.resnet_101_list = nn.ModuleList(list(self.resnet_101.children())[:-2]) # avgpool fc 제외

    def forward(self,x):
        x = self.resnet_101_list[0](x)  # 7 x 7 conv 64
        x = self.resnet_101_list[1](x)  # bn
        x = self.resnet_101_list[2](x)  # relu
        x = self.resnet_101_list[3](x)  # 3 x 3 maxpool

        x = self.resnet_101_list[4](x)  # layer 1
        c3 = x = self.resnet_101_list[5](x)  # layer 2
        c4 = x = self.resnet_101_list[6](x)  # layer 3
        c5 = x = self.resnet_101_list[7](c4)  # layer 4
        return [c3, c4, c5]

class ResNext_50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnext_50 = resnext50_32x4d(pretrained=pretrained)
        self.resnext_50_list = nn.ModuleList(list(self.resnext_50.children())[:-2])

    def forward(self,x):
        x = self.resnext_50_list[0](x)  # 7 x 7 conv 64
        x = self.resnext_50_list[1](x)  # bn
        x = self.resnext_50_list[2](x)  # relu
        x = self.resnext_50_list[3](x)  # 3 x 3 maxpool

        x = self.resnext_50_list[4](x)  # layer 1
        c3 = x = self.resnext_50_list[5](x)  # layer 2
        c4 = x = self.resnext_50_list[6](x)  # layer 3
        c5 = x = self.resnext_50_list[7](c4)  # layer 4
        return [c3, c4, c5]

class ResNext_101(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnext_101 = resnext101_32x8d(pretrained=pretrained)
        self.resnext_101_list = nn.ModuleList(list(self.resnext_101.children())[:-2])


    def forward(self,x):
        x = self.resnext_101_list[0](x)  # 7 x 7 conv 64
        x = self.resnext_101_list[1](x)  # bn
        x = self.resnext_101_list[2](x)  # relu
        x = self.resnext_101_list[3](x)  # 3 x 3 maxpool

        x = self.resnext_101_list[4](x)  # layer 1
        c3 = x = self.resnext_101_list[5](x)  # layer 2
        c4 = x = self.resnext_101_list[6](x)  # layer 3
        c5 = x = self.resnext_101_list[7](c4)  # layer 4
        return [c3, c4, c5]

if __name__ == "__main__":
    #base = ResNet_50(pretrained=True)
    #base = ResNet_101(pretrained=True)
    #base = ResNext_50(pretrained=True)
    base = ResNext_101(pretrained=True)
    image = torch.randn([2, 3, 800, 800])
    C3, C4, C5 = base(image)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)
    