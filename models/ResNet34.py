"""
author: Kwong
time: 2019/8/16 14:01

"""

import torch
import torch.nn as nn


class ResNet_34(nn.Module):
    def __init__(self, graph_shape, in_channel, out_channel):
        """
        :param graph_shape:  shape of image
        :param in_channel:   channels of input
        :param out_channel:  channels of output
        """
        super(ResNet_34, self).__init__()
        # preliminary
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        residual_block = [3, 4, 6, 3]
        # residual_block = [1, 2]
        self.feature_layer = self.feature_extract(residual_block)  # parameter is list

        # calculate input shape of linear layer
        # convolution    kernel size = 7, stride = 2, padding = 3
        sp = int((graph_shape - 7 + 2 * 3) / 2.0) + 1
        # maxpooling kernel size = 3, stride = 2, padding = 3
        sp = int((sp - 3 + 2 * 1) / 2) + 1
        # len(residual_block) - 1 times shortcut
        # convolution    kernel size = 1, stride = 2, padding = 0
        for i in range(len(residual_block) - 1):
            sp = int((sp - 1) / 2.0) + 1
        self.in_size = sp
        self.classifier = nn.Linear(in_features=64 * 2 ** (len(residual_block) - 1), out_features=out_channel)

    def feature_extract(self, block):
        layer = []
        for i, times in enumerate(block):
            if i != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_channels=64 * 2 ** (i - 1), out_channels=64 * 2 ** i
                              , kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(64 * 2 ** i)
                )
                layer.append(ResidualBlock(in_channel=64 * 2 ** (i - 1), out_channel=64 * 2 ** i,
                                           stride=2, shortcut=shortcut))
            for j in range(0 if i == 0 else 1, times):
                layer.append(ResidualBlock(in_channel=64 * 2 ** i, out_channel=64 * 2 ** i, stride=1))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.feature_layer(x)
        x = nn.AvgPool2d(self.in_size)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def conv_3x3(in_channels, out_channels, stride=1, padding=1):
    block = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3
                      , stride=stride, padding=padding, bias=False)
    return block


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            conv_3x3(in_channels=in_channel, out_channels=out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            conv_3x3(in_channels=out_channel, out_channels=out_channel),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = shortcut

    def forward(self, x):
        first = self.block(x)
        second = x if self.shortcut is None else self.shortcut(x)  # calculate residual block
        out = first + second
        return nn.ReLU(inplace=True)(out)


if __name__ == '__main__':
    # test
    graph = torch.rand([5, 3, 512, 512], requires_grad=True)
    net = ResNet_34(graph_shape=512, in_channel=3, out_channel=10)
    print(torch.argmax(net(graph)[0], dim=0))
    # print(net)