"""
author: Kwong
time: 2019/8/20 15:12

"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.models.resnet import resnet50

"""
    Reference from https://github.com/mehtanihar/pspnet/blob/master/models/network.py
"""


class PyramidPool(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size):
        super(PyramidPool, self).__init__()

        self.feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        sp = x.size()
        x = F.interpolate(self.feature(x), sp[2:], mode='bilinear', align_corners=False)
        return x


class PSPNet(nn.Module):
    def __init__(self, out_channel, pretrain=False):
        super(PSPNet, self).__init__()
        self.backbone = resnet50(pretrained=pretrain)
        self.pool5 = PyramidPool(2048, 512, 1)
        self.pool4 = PyramidPool(2048, 512, 2)
        self.pool3 = PyramidPool(2048, 512, 3)
        self.pool2 = PyramidPool(2048, 512, 6)
        self.final_layer = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channel, 1)
        )

    def forward(self, x):
        sp = x.size()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.final_layer(torch.cat([
            x, self.pool5(x), self.pool4(x), self.pool3(x), self.pool2(x)
        ], dim=1))
        x = F.interpolate(x, sp[2:], mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    graph = torch.rand([5, 3, 128, 128], requires_grad=True)
    net = PSPNet(out_channel=6)
    print(torch.argmax(net(graph)[0], dim=0))