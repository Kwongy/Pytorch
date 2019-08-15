"""
author: Kwong
time: 2019/8/15 12:10

"""
import torch
import torch.nn as nn


def conv_relu(in_channel, out_channel, kernel_size, stride=1, padding=0):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )
    return block


class GoogLeNet(nn.Module):
    def feature_extract(self, in_channel):
        block = nn.Sequential(
            conv_relu(in_channel=in_channel, out_channel=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            conv_relu(in_channel=64, out_channel=64, kernel_size=1),
            conv_relu(in_channel=64, out_channel=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            Inception(in_channel=192, out1_1=64, out2_1=96, out2_3=128, out3_1=16, out3_5=32, out4_1=32),
            Inception(in_channel=256, out1_1=128, out2_1=128, out2_3=192, out3_1=32, out3_5=96, out4_1=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            Inception(in_channel=480, out1_1=192, out2_1=96, out2_3=208, out3_1=16, out3_5=48, out4_1=64),
            Inception(in_channel=512, out1_1=160, out2_1=112, out2_3=224, out3_1=24, out3_5=64, out4_1=64),
            Inception(in_channel=512, out1_1=128, out2_1=128, out2_3=256, out3_1=24, out3_5=64, out4_1=64),
            Inception(in_channel=512, out1_1=112, out2_1=144, out2_3=288, out3_1=32, out3_5=64, out4_1=64),
            Inception(in_channel=528, out1_1=256, out2_1=160, out2_3=320, out3_1=32, out3_5=128, out4_1=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            Inception(in_channel=832, out1_1=256, out2_1=160, out2_3=320, out3_1=32, out3_5=128, out4_1=128),
            Inception(in_channel=832, out1_1=384, out2_1=182, out2_3=384, out3_1=48, out3_5=128, out4_1=128),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        )
        return block

    def classifier(self, in_size, out_channel):
        block = nn.Sequential(
            nn.Linear(in_features=in_size * in_size * 1024, out_features=out_channel),
        )
        return block

    def __init__(self, graph_shape, in_channel, out_channel):
        super(GoogLeNet, self).__init__()
        # Convolution layer
        self.feature_layer = self.feature_extract(in_channel=in_channel)

        # calculate input shape of Linear layer
        sp = graph_shape
        sp = int((sp - 7 + 2 * 3) / 2.0) + 1  # convolution layer 1
        for i in range(4):      # 4 MaxPooling layer
            sp = int((sp - 3) / 2.0) + 1
        sp = sp - 2 + 1 # 1 AvgPooling

        # fully Linear layer
        self.classifier_layer = self.classifier(in_size=sp, out_channel=out_channel)

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_layer(x)
        return x


# Inception Module
class Inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(Inception, self).__init__()
        self.struct1 = conv_relu(in_channel=in_channel, out_channel=out1_1, kernel_size=1)
        self.struct2 = nn.Sequential(
            conv_relu(in_channel=in_channel, out_channel=out2_1, kernel_size=1),
            conv_relu(in_channel=out2_1, out_channel=out2_3, kernel_size=3, padding=1)
        )
        self.struct3 = nn.Sequential(
            conv_relu(in_channel=in_channel, out_channel=out3_1, kernel_size=1),
            conv_relu(in_channel=out3_1, out_channel=out3_5, kernel_size=5, padding=2)
        )
        self.struct4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_relu(in_channel=in_channel, out_channel=out4_1, kernel_size=1)
        )

    def forward(self, x):
        part1 = self.struct1(x)
        part2 = self.struct2(x)
        part3 = self.struct3(x)
        part4 = self.struct4(x)
        return torch.cat([part1, part2, part3, part4], dim=1)


if __name__ == '__main__':
    graph = torch.rand([5, 3, 1024, 1024], requires_grad=True)
    net = GoogLeNet(graph_shape=1024, in_channel=3, out_channel=10)
    # print(net(graph))
    print(torch.argmax(net(graph)[0], dim=0))