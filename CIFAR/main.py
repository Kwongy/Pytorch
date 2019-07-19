#coding = utf-8

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from lenet5 import Lenet5
from torch import nn, optim


def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar', True, transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]), download= True)
    cifar_train = DataLoader(cifar_train, batch_size = batch_size, shuffle = True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)


    x, label = iter(cifar_train).next()
    print('x : ', x.shape, 'label : ', label.shape)

    # return

    # device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = Lenet5()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    print(model)

    for epoch in range(1000):

        model.train()
        for batch_idx, (x, label) in enumerate(cifar_train):
            # x, label = x.to(device), label.to(device)
            # x:[batch, 3, 32, 32] label : [b]
            logits = model(x)
            # logits:[b, 10]
            loss = criteon(logits, label)
            # loss: tensor scalar

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
        # test
            total_correct, total_num = 0, 0
            for x, label in cifar_test:
                # x, label = x.to(device), label.to(device)
                # [batch, 10]
                logits = model(x)
                # [batch]
                pred = logits.argmax(dim = 1)
                # [b] vs [b]
                total_correct += torch.eq(pred, label).float().sum()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()