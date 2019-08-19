"""
author: Kwong
time: 2019/8/15 10:08

"""

import torch
import torch.nn as nn


class VGG_16(nn.Module):
    def __init__(self, graph_shape, in_channel, out_channel):
        super(VGG_16, self).__init__()
        # Convolution layer
        self.conv_layer = nn.Sequential(
            self.conv_3x3(in_channels=in_channel, out_channels=64),
            self.conv_3x3(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_3x3(in_channels=64, out_channels=128),
            self.conv_3x3(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_3x3(in_channels=128, out_channels=256),
            self.conv_3x3(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_3x3(in_channels=256, out_channels=512),
            self.conv_3x3(in_channels=512, out_channels=512),
            self.conv_3x3(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_3x3(in_channels=512, out_channels=512),
            self.conv_3x3(in_channels=512, out_channels=512),
            self.conv_3x3(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # calculate input shape of Linear layer
        sp = graph_shape
        for i in range(5):
            sp = int((sp - 2) / 2.) + 1

        # fully connection layer
        self.conn_layer = self.fully_conn(in_size=sp, out_channel=out_channel)

    def conv_3x3(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def fully_conn(self, in_size, out_channel):
        block = nn.Sequential(
            nn.Linear(in_features=in_size * in_size * 512, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=out_channel)
        )
        return block

    def forward(self, x):
        x = self.conv_layer(x)
        # change multi-channels to one channel
        x = x.view(x.size(0), -1)
        x = self.conn_layer(x)
        return x


if __name__ == '__main__':
    # test
    graph = torch.rand([5, 3, 512, 512], requires_grad=True)
    net = VGG_16(graph_shape=512, in_channel=3, out_channel=10)
    print(torch.argmax(net(graph)[0], dim=0))
