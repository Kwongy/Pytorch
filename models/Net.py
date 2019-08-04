"""
author: Kwong
time: 2019/8/4 13:12

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# inherit torch.nn.Module
class studyNet(nn.Module):
    # define neural network
    def __init__(self):
        super(studyNet, self).__init__()
        # convolution layer
        # represent : input_channel = 1， output_channel = 10， kernel_size = 3
        self.conv1 = nn.Conv2d(1, 10, 3)
        # linear layer
        # represent : input_channel =  2250(base of function forward), output = 10
        self.fc1 = nn.Linear(2250, 10)

    # define network structure
    # input x : tensor [1, 1, 32, 32] (size of image) [batch, channel, height, width]
    def forward(self, x):
        x = self.conv1(x)   # convolution
        # print(x.size())   # torch.Size([1, 10, 30, 30])
        x = F.relu(x)       # activation function relu
        x = F.max_pool2d(x, (2, 2)) # max pooling , kernel_size = 2 * 2
        # print(x.size())     # torch.Size([1, 10, 15, 15])
        x = F.relu(x)       # activation function relu
        x = x.view(x.size()[0], -1)    # turn each image into one dimension
        # -1 represent self-adapton , program can compute itself
        # print(x.size()) # torch.Size([1, 2250])
        x = self.fc1(x)  # need to define size in __init__ function self.fc1 = nn.Linear(2250)
        return x


if __name__ == '__main__':
    net = studyNet()
    # print(net)
    # studyNet(
    #     (conv1): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
    #     (fc1): Linear(in_features=2250, out_features=10, bias=True)
    # )

