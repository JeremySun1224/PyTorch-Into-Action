# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/7 -*-


import torch
import torchvision
import time
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

print('Torchvision Version:', torchvision.__version__)

# 数据
data_dir = './hymenoptera_data'
model_name = 'resnet'
num_classes = 2
batch_size = 32
num_epochs = 15
feature_extract = True
input_size = 224

# 读入数据
all_imgs = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose([
    transforms.RandomResizedCrop(input_size),  # 把每张图片变成resnet需要输入的维度224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]))
loader = DataLoader(all_imgs, batch_size=batch_size, shuffle=True)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']
}

dataloaders_dict = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img = next(iter(dataloaders_dict['valid']))[0]
print(img.shape)


