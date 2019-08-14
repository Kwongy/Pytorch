"""
author: Kwong
time: 2019/8/14 13:27

"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
        graph_shape = shape of image
        in_channel = input channel
        out_channel = class of image
        input image : [batch, channel, height shape, width shape]
                note: height = width
    """
    def __init__(self, graph_shape, in_channel, out_channel):
        super(AlexNet, self).__init__()
        # feature_extract
        self.feature_layer = self.feature_extract(in_channel)

        # calculate input shape
        sp = int((graph_shape - 11 + 4) / 4) + 1
        sp = int((sp - 3) / 2) + 1
        sp = int((sp - 5 + 4) / 1) + 1
        sp = int((sp - 3) / 2) + 1
        # for i in range(3):
        #     sp = int((sp - 3 + 2) / 1) + 1
        sp = int((sp - 3) / 2) + 1

        # classifier
        self.classifier_layer = self.classifier(in_size=sp, out_channel=out_channel)

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_layer(x)
        return x

    def feature_extract(self, in_channel):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        return block

    def classifier(self, in_size, out_channel):
        block = nn.Sequential(
            nn.Linear(in_size * in_size * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_channel)
        )
        return block


if __name__ == '__main__':
    graph = torch.rand([5, 3, 256, 256], requires_grad=True)
    net = AlexNet(graph_shape=256, in_channel=3, out_channel=10)
    print(torch.argmax(net(graph)[0], dim=0))