"""
author: Kwong
time: 2019/9/15 20:15

"""

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.segmentation.segmentation import fcn_resnet101
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def pre_fcn_resnet101(in_channel, out_channel):
    model = fcn_resnet101(pretrained=False, progress=False)
    url = "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth"  # COCO
    model_dict = model.state_dict()
    pretrained_dict = model_zoo.load_url(url, progress=False)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = FCNHead(2048, out_channel)
    return model


def pre_deeplabv3_resnet101(in_channel, out_channel):
    model = deeplabv3_resnet101(pretrained=False)
    url = 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
    model_dict = model.state_dict()
    pretrained_dict = model_zoo.load_url(url, progress=False)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = DeepLabHead(2048, out_channel)
    return model


if __name__ == "__main__":
    graph = torch.rand([5, 4, 256, 256], requires_grad=True)
    model = pre_fcn_resnet101(in_channel=4, out_channel=2)
    # print(model)
    print(model(graph)['out'])
