#!/usr/bin/env python
# coding: utf-8


import torch
import random
import copy
import torchtext
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torchtext.vocab import Vectors

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

random.seed(1224)
np.random.seed(1224)
torch.manual_seed(1224)
if USE_CUDA:
    torch.cuda.manual_seed(1224)

NUM_EPOCHS = 1
BATCH_SIZE = 64
GRAD_CLIP = 5.0
HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path='../02词向量简介/text8', text_field=TEXT,
    train='text8.train.txt', validation='text8.dev.txt', test='text8.test.txt')

# # 创建Vocabulary


TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    datasets=(train, val, test), batch_size=BATCH_SIZE, device=DEVICE,
    bptt_len=50, repeat=False, shuffle=True)

it = iter(train_iter)
batch = next(it)


# 定义模型

# - 继承nn.Module
# - 初始化__init__()函数
# - 定义forward()函数
# - 其余可以根据模型需要定义相关函数


class RNNModel(nn.Module):
    def __init__(self, rnn_type, n_token, n_input, n_hidden, n_layers, dropout=0.5):
        """
        模型包含以下层:
            - 词嵌入层
            - 一个循环网络层（RNN, LSTM, GRU）
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization        
        """
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_embeddings=n_token, embedding_dim=n_input)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_input, n_hidden, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for '--model' was suppiled, options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']")
            self.rnn = nn.RNN(n_input, n_hidden, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(n_hidden, n_token)
        self.init_weights()
        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        """
        Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.n_layers, batch_size, self.n_hidden), requires_grad=requires_grad),
                    weight.new_zeros((self.n_layers, batch_size, self.n_hidden), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.n_layers, batch_size, self.n_hidden), requires_grad=requires_grad)


# - 初始化一个RNN模型

VOCAB_SIZE = len(TEXT.vocab)
model = RNNModel(rnn_type='LSTM', n_token=VOCAB_SIZE, n_input=EMBEDDING_SIZE, n_hidden=HIDDEN_SIZE, n_layers=2,
                 dropout=0.5)
if USE_CUDA:
    model = model.cuda()


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history"""
    if isinstance(h, torch.Tensor):
        return h.detach()  # detach()一定要加括号
    else:
        return tuple(repackage_hidden(v) for v in h)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)


def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(batch_size=BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())
    loss = total_loss / total_count
    model.train()
    return loss


# - 模型训练
#     - 模型一般需要训练若干个epoch
#     - 每个epoch我们都把所有的数据分成若干个batch
#     - 把每个batch的输入湖人输出都包装成cuda tensor
#     - forward pass，通过输入的句子预测每个单词的下一个单词
#     - 用模型的预测和正确的下一个单词计算cross entropy loss
#     - 清空模型当前的gradient
#     - backward pass
#     - gradient clipping，防止梯度爆炸
#     - 更新模型参数
#     - 每隔一定的iteration输出模型在当前iteration的loss以及在验证集上做模型的评估


val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(batch_size=BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        hidden = repackage_hidden(h=hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 1000 == 0:
            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
        if i % 10000 == 0:
            val_loss = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_loss < min(val_losses):
                print(f'Best model, val loss: {val_loss}')
                torch.save(model.state_dict(), 'lm_best.pth')
            else:
                scheduler.step()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            val_losses.append(val_loss)
