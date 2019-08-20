"""
author: Kwong
time: 2019/8/20 9:26

"""
import torch
import torch.nn as nn


def conv_3x3(in_channel, out_channel):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )
    return block


class encode_2x(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(encode_2x, self).__init__()
        self.conv1 = conv_3x3(in_channel=in_channel, out_channel=out_channel)
        self.conv2 = conv_3x3(in_channel=out_channel, out_channel=out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        sp = x.size()
        x, index = self.maxpool(x)
        return x, index, sp


class encode_3x(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(encode_3x, self).__init__()
        self.conv1 = conv_3x3(in_channel=in_channel, out_channel=out_channel)
        self.conv2 = conv_3x3(in_channel=out_channel, out_channel=out_channel)
        self.conv3 = conv_3x3(in_channel=out_channel, out_channel=out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        sp = x.size()
        x, index = self.maxpool(x)
        return x, index, sp


class decode_2x(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decode_2x, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv2 = conv_3x3(in_channel=in_channel, out_channel=in_channel)
        self.conv1 = conv_3x3(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x, index, sp):
        x = self.unpool(input=x, indices=index, output_size=sp)
        x = self.conv2(x)
        x = self.conv1(x)
        return x


class decode_3x(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decode_3x, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv3 = conv_3x3(in_channel=in_channel, out_channel=in_channel)
        self.conv2 = conv_3x3(in_channel=in_channel, out_channel=in_channel)
        self.conv1 = conv_3x3(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x, index, sp):
        x = self.unpool(input=x, indices=index, output_size=sp)
        x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x


class SegNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SegNet, self).__init__()
        self.encode1 = encode_2x(in_channel, 64)
        self.encode2 = encode_2x(64, 128)
        self.encode3 = encode_3x(128, 256)
        self.encode4 = encode_3x(256, 512)
        self.encode5 = encode_3x(512, 512)
        self.decode5 = decode_3x(512, 512)
        self.decode4 = decode_3x(512, 256)
        self.decode3 = decode_3x(256, 128)
        self.decode2 = decode_2x(128, 64)
        self.decode1 = decode_2x(64, out_channel)

    def forward(self, x):
        x, index1, sp1 = self.encode1(x)
        x, index2, sp2 = self.encode2(x)
        x, index3, sp3 = self.encode3(x)
        x, index4, sp4 = self.encode4(x)
        x, index5, sp5 = self.encode5(x)

        x = self.decode5(x, index5, sp5)
        x = self.decode4(x, index4, sp4)
        x = self.decode3(x, index3, sp3)
        x = self.decode2(x, index2, sp2)
        x = self.decode1(x, index1, sp1)

        return x


if __name__ == '__main__':
    # test
    graph = torch.rand([5, 3, 128, 128], requires_grad=True)
    net = SegNet(in_channel=3, out_channel=2)
    print(torch.argmax(net(graph)[0], dim=0))
