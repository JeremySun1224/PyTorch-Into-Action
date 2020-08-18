# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/2 -*-


import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import random
from torchtext import data
from torchtext import datasets

# 准备数据
SEED = 1224
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = ['cuda:0' if torch.cuda.is_available() else 'cpu']
BATCH_SIZE = 64
TEXT = data.Field(tokenize='spacy')
LABEL = data.Field(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 建立vocabulary
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

# 创建iterator
train_iter, valid_iter, test_iter = data.BucketIterator(
    dataset=(train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=DEVICE
)


# Word Averaging Model
class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAVGModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=pad_idx)
        self.linear = nn.Linear(in_features=embedding_size, out_features=output_size)

    def forward(self):
        embedded = self.embed(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze()
        return self.linear(pooled)


VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = WordAVGModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, output_size=OUTPUT_SIZE, pad_idx=PAD_IDX)
print(model)