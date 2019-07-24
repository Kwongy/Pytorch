"""
author: Kwong
time: 2019/7/24 19:49

"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import sys
sys.path.append("..")
from LoadData import MyDataset
from tensorboardX import SummaryWriter
from datetime import datetime

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 1

train_txt_path = '../../../datasets/train.txt'



# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path)
# valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

for i, data in enumerate(train_loader):
    inputs, labels = data
    print(inputs.shape, labels, len(labels))

# img = cv2.imread("F:/datasets/train/img_2017/image_2017_960_960_1.tif")
# print(img.shape)
# print(img)
# cv2.imshow("image", img)
# cv2.waitKey(0)