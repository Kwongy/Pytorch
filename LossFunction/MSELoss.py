"""
author: Kwong
time: 2019/8/5 9:05

"""

from Pytorch.models.Net import studyNet
import torch
import torch.nn as nn

"""
    MSELoss :  [/sum_{i = 0}^{n - 1} (out_i - target_i)^2] / n
"""

net = studyNet()
graph = torch.randn([1, 1, 32, 32])
out = net(graph)
net.zero_grad()
out.backward(torch.ones(1, 10))     # Automatic back propagation
criterion = nn.MSELoss()
y = torch.randn([1, 10])
loss = criterion(out, y)
singleLoss = nn.MSELoss(reduction='none')
averageLoss = nn.MSELoss(reduction='mean')
sumLoss = nn.MSELoss(reduction='sum')
loss1 = singleLoss(out, y)
loss2 = averageLoss(out, y)
loss3 = sumLoss(out, y)

print('no parameters: {}'.format(loss.item()))
print('reduction = none, loss of every dimension :\n{}'.format(loss1))
print('reduction = mean，\t average :\t{}'.format(loss2))
print('reduction = sum ，\t sum :\t{}'.format(loss3))

cnt = 0
for i in range(y.size()[1]):
    cnt += pow(y[0, i] - out[0, i], 2)
print('calculate separately : {}'.format(cnt))