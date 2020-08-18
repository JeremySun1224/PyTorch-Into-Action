# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/7/26 -*-

import torch
import spacy
import torch.nn as nn
from torchtext import data
from torchtext import datasets

SEED = 1224
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)