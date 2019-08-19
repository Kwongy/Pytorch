"""
author: Kwong
time: 2019/8/19 8:48

"""

import torch
import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.segmentation import FCN


# rewrite vgg
class vggNet(VGG):
    def __init__(self, in_channel, model='vgg16'):
        super().__init__(make_layers(cfg[model], in_channels=in_channel))
        del self.classifier
        # Freezing parameters
        for parameter in super().parameters():
            parameter.requires_grad = False

        self.ranges = ranges[model]

    def forward(self, x):
        ls = []
        for i, (first, second) in enumerate(self.ranges):
            for layer in range(first, second):
                x = self.features[layer](x)
            ls.append(x)
        return ls


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


def make_layers(cfg, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class FCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCN, self).__init__()
        self.backbone = vggNet(in_channel)
        self.deconv1 = self.convTranspose(512, 512)
        self.deconv2 = self.convTranspose(512, 256)
        self.deconv3 = self.convTranspose(256, 128)
        self.deconv4 = self.convTranspose(128, 64)
        self.deconv5 = self.convTranspose(64, 32)
        self.classifier = nn.Conv2d(32, out_channel, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.deconv1(x5) + x4
        x = self.deconv2(x) + x3
        x = self.deconv3(x) + x2
        x = self.deconv4(x) + x1
        x = self.deconv5(x)
        x = self.classifier(x)
        return x

    def convTranspose(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                               stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )
        return block


if __name__ == '__main__':
    graph = torch.rand([5, 6, 128, 128], requires_grad=True)
    net = FCN(in_channel=6, out_channel=10)
    # print(net)
    import cv2
    import numpy as np

    pred = net(graph)
    predictions = torch.argmax(pred[0], dim=0).detach().numpy().astype(np.uint8)

    cv2.imshow('img', predictions)
    cv2.waitKey(0)
