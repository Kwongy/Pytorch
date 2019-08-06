"""
author: Kwong
time: 2019/8/6 13:12

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Pytorch.models.Net import studyNet
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt


learningrate = 0.01
batchsize = 10
EPOCH = 12

x = torch.randn([10, 1, 32, 32])
y = torch.from_numpy(np.random.randint(low = 0, high=9, size= 10)).type(torch.LongTensor)

dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=dataset,
    batch_size= batchsize,
    shuffle=True,
    num_workers=5
)

net_SGD = studyNet()
net_Momentum = studyNet()
net_RMSProp = studyNet()
net_Adam= studyNet()
nets = [net_SGD,net_Momentum,net_RMSProp,net_Adam]

opt_SGD = optim.SGD(net_SGD.parameters(), lr=learningrate)
opt_Momentum = optim.SGD(net_Momentum.parameters(), lr=learningrate, momentum=0.8)# 是SGD的改进，加了动量效果
opt_RMSProp = optim.RMSprop(net_RMSProp.parameters(), lr=learningrate, alpha=0.9)
opt_Adam= optim.Adam(net_Adam.parameters(), lr=learningrate, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

loss_function = nn.CrossEntropyLoss()
Loss = [[],[],[],[]]

if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):
            v_x = Variable(x)
            v_y = Variable(y)

            for net, opt, l_his in zip(nets, optimizers, Loss):
                output = net(v_x)  # get_out for every net
                loss = loss_function(output, v_y)  # compute loss for every net
                opt.zero_grad()
                loss.backward()
                opt.step()  # apply gradient
                l_his.append(loss)  # loss recoder

    print(Loss)
    labels = ['SGD', 'Momentum', 'RMSProp', 'Adam']
    for i, l_his in enumerate(Loss):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

