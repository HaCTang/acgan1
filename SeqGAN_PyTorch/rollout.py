# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, lstm, update_rate, pad_num):
        super(Rollout, self).__init__()

        self.lstm = lstm
        # self.own_model = copy.deepcopy(lstm)
        self.update_rate = update_rate
        self.pad_num = pad_num
        
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.use_cuda = self.lstm.use_cuda
        self.sequence_length = self.lstm.sequence_length
        self.start_token = self.lstm.start_token
        self.learning_rate = self.lstm.learning_rate

        self.emb = self.lstm.emb
        self.class_emb = self.lstm.class_emb

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # self.init_params()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def get_reward(self, x, num, discriminator, class_labels):
        """
        Args:
            x : (batch_size, seq_len) input data (generated sequences)
            num : roll-out number
            discriminator : discrimanator model
            class_labels: (batch_size,) ground truth class labels
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                
                # Get both real/fake prediction and class prediction from discriminator
                real_fake_pred, class_pred = discriminator(samples)
                real_fake_pred = real_fake_pred.cpu().data.numpy()
                class_pred = class_pred.cpu().data.numpy()
                
                # Compute rewards based on real/fake prediction
                real_fake_reward = real_fake_pred[:, 0]  # Assuming 0 index for real
                # Optionally, include class prediction accuracy in reward
                class_reward = np.sum(np.argmax(class_pred, axis=1) == class_labels.cpu().numpy()) / float(batch_size)
                total_reward = real_fake_reward + class_reward  # Combine both rewards
                if i == 0:
                    rewards.append(total_reward)
                else:
                    rewards[l-1] += total_reward

            # for the last token
            real_fake_pred, class_pred = discriminator(x)
            real_fake_pred = real_fake_pred.cpu().data.numpy()
            class_pred = class_pred.cpu().data.numpy()
            
            real_fake_reward = real_fake_pred[:, 0]  # Real/fake reward
            class_reward = np.sum(np.argmax(class_pred, axis=1) == class_labels.cpu().numpy()) / float(batch_size)
            
            total_reward = real_fake_reward + class_reward  # Combine rewards for the last token
    
            if i == 0:
                rewards.append(total_reward)
            else:
                rewards[seq_len-1] += total_reward
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]