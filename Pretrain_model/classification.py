"""
author: Kwong
time: 2019/9/15 19:13

"""
import torch.nn as nn
import torch
from torchvision.models.densenet import densenet201
from torchvision.models.resnet import resnet152


def pre_densenet201(in_channel=3, out_channel=1000):
    model = densenet201(pretrained=True, progress=False)
    model.features.conv0 = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, out_channel, True)
    return model


def pre_resnet152(in_channel=3, out_channel=1000):
    model = resnet152(pretrained=True, progress=False)
    model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_channel)
    return model


if __name__ == "__main__":
    graph = torch.rand([5, 4, 256, 256], requires_grad=True)
    model = pre_resnet152(in_channel=4, out_channel=2)
    print(model)
    # print(model(graph))