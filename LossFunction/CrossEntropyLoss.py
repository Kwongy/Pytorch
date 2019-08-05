"""
author: Kwong
time: 2019/8/5 14:37

"""

from Pytorch.models.Net import studyNet
import torch
import torch.nn as nn
import numpy as np
import math

net = studyNet()
graph = torch.randn([10, 1, 32, 32])
out = net(graph)
# net.zero_grad()
# out.backward(torch.ones(1, 10))

"""
    CrossEntropyLoss : aim at a single objective problem
    
    input : data [batch, channel, x1, x2 ,..]
            target [0, Class_num - 1]  (classification of objective)
            
    loss(x, class) = weight[class](-x[class] + log\sum{exp(x[j])})
"""


target = torch.from_numpy(np.random.randint(low = 0, high=9, size= 10)).type(torch.LongTensor)

print('out : {} \n target : {}'.format(out.detach().numpy(), target.detach().numpy()))
print('--------- calculate loss of index zero by myself ---------')

output = out.detach().numpy()
x0_class = output[0, target[0].detach().numpy()]
sumlog = 0
for i in range(output.shape[1]):
    sumlog += pow(math.e, output[0, i])
sumlog = math.log(sumlog)
loss_0 = -x0_class + sumlog
print('Manual calculation result : loss_0 = {}'.format(loss_0))

print("--------- calculate loss by function ---------")
weight = torch.from_numpy(np.random.randn(10)).float()
loss_function1 = nn.CrossEntropyLoss(reduction='mean')
loss_function2 = nn.CrossEntropyLoss(reduction='none')
loss_function3 = nn.CrossEntropyLoss(weight=weight, reduction='none')
loss_function4 = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

lossbyf1 = loss_function1(out, target)
lossbyf2 = loss_function2(out, target)
lossbyf3 = loss_function3(out, target) # loss_i = loss[i] * weight[i]
lossbyf4 = loss_function4(out, target) # ignore loss of index 0, Calculate the rest

print(" calculate the average loss :\n {}".format(lossbyf1))
print(" calculate each loss separately :\n {}".format(lossbyf2))
print(" calculate weighted loss : \n weight : {} \n loss : {}".format(weight, lossbyf3))
print(" calculate ignore_index 0 :\n {} \n target : {}".format(lossbyf4, target))





