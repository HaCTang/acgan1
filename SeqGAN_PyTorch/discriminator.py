# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Highway >> Softmax (Real/Fake & Class)
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)

        # Linear layers for Real/Fake classification and Class classification
        self.lin_real_fake = nn.Linear(sum(num_filters), 1)  # Binary output for real/fake
        self.lin_class = nn.Linear(sum(num_filters), num_classes)  # Class output
        # self.lin = nn.Linear(sum(num_filters), num_classes)
        # self.softmax = nn.LogSoftmax()

        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        # Apply dropout
        pred = self.dropout(pred)
        # Real/Fake classification
        real_fake_output = torch.sigmoid(self.lin_real_fake(pred))  # Binary classification (real/fake)
        # Class prediction
        class_output = self.lin_class(pred)  # Class classification (multiclass)
        return real_fake_output, class_output

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
