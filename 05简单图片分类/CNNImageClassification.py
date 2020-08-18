# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/6 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 加载数据集
mnist_data = datasets.MNIST('./mnist_data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
# print(mnist_data)
# print(mnist_data[0][1])  # 数字
# print(mnist_data[1][0].shape)

data = [d[0].data.cpu().numpy() for d in mnist_data]
MEAN = np.mean(data)
STD = np.std(data)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNINGRATE = 0.01
MOMENTUM = 0.5
NUM_EPOCHS = 10

train_loader = DataLoader(
    dataset=datasets.MNIST('./mnist_data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(MEAN,), std=(STD,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True
)

test_loader = DataLoader(
    dataset=datasets.MNIST('./mnist_data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(MEAN,), std=(STD,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True
)


# 定义基于ConvNet的简单神经网络
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 28 * 28 -> (28 + 1 - 5), namely, 24 * 24
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 24 * 24 -> (24 + 1 - 5), namely, 20 * 20
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  # x = [N, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [N, 20, 24, 24]
        x = F.max_pool2d(x, 2, 2)  # [N, 20, 12, 12]
        x = F.relu(self.conv2(x))  # [N, 50, 8, 8]
        x = F.max_pool2d(x, 2, 2)  # [N, 50, 4, 4]
        x = x.view(-1, 4 * 4 * 50)  # [N, 4 * 4 * 50]
        x = F.relu(self.fc1(x))  # [N, 4 * 4 * 50] * [4 * 4 * 50, 500] = [N, 500]
        x = self.fc2(x)  # [N, 500] * [500, 10] = [N, 10]
        return F.log_softmax(x, dim=1)  # 带log的softmax分类，每张图片返回10个概率


cnn_model = CNNModel()
if torch.cuda.is_available():
    cnn_model = cnn_model.to(DEVICE)
optimizer = optim.SGD(params=cnn_model.parameters(), lr=LEARNINGRATE, momentum=MOMENTUM)


# 定义训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.nll_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f'Train Epoch: {epoch}, Iteration: {idx}, Loss: {loss.item()}')


# 定义测试
def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)  # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)  # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print(f'Test loss: {total_loss}, Accuracy: {acc}')
    model.train()
    return total_loss


def main():
    best_mnist_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train(model=cnn_model, device=DEVICE, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        test(model=cnn_model, device=DEVICE, test_loader=test_loader)
        total_loss = test(model=cnn_model, device=DEVICE, test_loader=test_loader)
        if total_loss < best_mnist_loss:
            torch.save(cnn_model.state_dict(), 'best_mnist_model.pth')


if __name__ == '__main__':
    main()
