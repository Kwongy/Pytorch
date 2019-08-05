"""
author: Kwong
time: 2019/8/4 9:14
"""
from Pytorch.models.Net import studyNet
import torch
import torch.nn as nn

net = studyNet()
graph = torch.randn([1, 1, 32, 32])
out = net(graph)
net.zero_grad()
out.backward(torch.ones(1, 10))     # Automatic back propagation
y = torch.randn((1, 10))
print('y : {}'.format(y))
print('out : {}'.format(out))
criterion = nn.L1Loss()
reduce_False = nn.L1Loss(reduction='none')
size_average_True = nn.L1Loss(reduction='mean')
size_average_False = nn.L1Loss(reduction='sum')
loss = criterion(out, y)
o_0 = reduce_False(out, y)
o_1 = size_average_True(out, y)
o_2 = size_average_False(out, y)
print('no parameters: {}'.format(loss.item()))
print('reduction = none, loss of every dimension :\n{}'.format(o_0))
print('reduction = mean，\t average :\t{}'.format(o_1))
print('reduction = sum ，\t sum :\t{}'.format(o_2))
cnt = 0
for i in range(y.size()[1]):
    cnt += abs(y[0, i] - out[0, i])
print('calculate separately : {}'.format(cnt))
