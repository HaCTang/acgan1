# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Generator(nn.Module):
    """Generator """
    def __init__(self, num_emb, batch_size, emb_dim, class_emb_dim, hidden_dim, num_classes, use_cuda,
                 sequence_length, start_token, learning_rate=0.001, reward_gamma=0.95, grad_clip=5):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.class_emb_dim = class_emb_dim
        self.pretrain_emb_dim = emb_dim + class_emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.start_token = torch.tensor([start_token] * self.batch_size, dtype=torch.long)
        self.learning_rate = learning_rate
        self.reward_gamma = reward_gamma
        self.grad_clip = grad_clip
        self.temperature = 1.0

        self.pretrain_emb = nn.Embedding(num_emb, self.pretrain_emb_dim)
        self.emb = nn.Embedding(num_emb, emb_dim)
        # nn.init.normal_(self.emb.weight, std=0.1)
        self.class_emb = nn.Embedding(num_classes, class_emb_dim) # Class embedding

        self.lstm = nn.LSTM(emb_dim + class_emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.init_params()

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0, 0.01)

    def forward(self, x, class_label, hidden, label_input=False):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
            class_label: (batch_size, ), class label for the sequences
        """
        x = x.to(self.emb.weight.device)
        # class_label = class_label.to(self.class_emb.weight.device)
        hidden = tuple(h.to(self.emb.weight.device) for h in hidden)
        x_emb = self.emb(x) # (batch_size, seq_len, emb_dim)

        if label_input:
            class_label = class_label.to(self.class_emb.weight.device)
            class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
            class_emb = class_emb.expand(-1, x.size(1), -1)       # (batch_size, seq_len, emb_dim)
            combined_emb = torch.cat([x_emb, class_emb], dim=-1)  # (batch_size, seq_len, emb_dim + class_emb_dim)
        else:
            batch_size, seq_len, _ = x_emb.size()
            zero_class_emb = torch.zeros(batch_size, seq_len, self.class_emb_dim, device=x.device)
            combined_emb = torch.cat([x_emb, zero_class_emb], dim=-1)  # (batch_size, seq_len, emb_dim + class_emb_dim)


        # # # Embedding for class labels and expand to match sequence length
        # class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
        # class_emb = class_emb.expand(-1, x.size(1), -1)       # (batch_size, seq_len, emb_dim)
        # # # Concatenate token embeddings with class embeddings
        # combined_emb = torch.cat([emb, class_emb], dim=-1)  # (batch_size, seq_len, emb_dim + class_emb_dim)

        output, hidden = self.lstm(combined_emb, hidden)
        logits = self.lin(output.contiguous().view(-1, self.hidden_dim))
        logits = logits.view(x.size(0), x.size(1), -1)

        # class_logits = self.classifier(output[:, -1, :])
        return logits, x_emb, hidden
        # return logits.view(x.size(0), x.size(1), -1), class_logits, hidden

    def pretrain_loss(self, x):
        """
        Calculates the pretraining loss.
        x: (batch_size, seq_len)
        logits: (batch_size, seq_len, vocab_size)
        """
        hidden = self.init_hidden(x.size(0))
        x = x.to(self.emb.weight.device)
        hidden = tuple(h.to(self.emb.weight.device) for h in hidden)
        logits, _, _ = self.forward(x, torch.zeros(x.size(0), dtype=torch.long), hidden) # (batch_size, seq_len, vocab_size)
        labels = x.detach().clone()
        
        # first_pad_token = (labels == self.num_emb - 1).to(torch.long)
        # first_pad_token = torch.argmax(first_pad_token, dim=-1)
        # for i in range(first_pad_token.size(0)):
        #     labels[i, first_pad_token[i]+1:] = -100
        
        labels = labels[:, 1:].contiguous().view(-1)  # (batch_size * seq_len)

        logits = logits[:, :-1, :].contiguous().view(-1, self.num_emb)  # (batch_size * seq_len, vocab_size)
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def pretrain_step(self, x):
        """
        Performs a pretraining step without class label.
        """
        self.optimizer.zero_grad()
        loss = self.pretrain_loss(x)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def train_loss(self, x, class_label, rewards):
        """
        Calculates the generator loss with rewards.
        """
        hidden = self.init_hidden(x.size(0))
        x = x.to(self.emb.weight.device)
        class_label = class_label.to(self.class_emb.weight.device)
        rewards = rewards.to(self.emb.weight.device)
        hidden = tuple(h.to(self.emb.weight.device) for h in hidden)
        logits, _, _ = self.forward(x, class_label, hidden, label_input=True)
        
        labels = x.detach().clone()
        labels = labels[:, 1:].contiguous().view(-1)  # (batch_size * (seq_len - 1))
        logits = logits[:, :-1, :].contiguous().view(-1, self.num_emb)  # (batch_size * (seq_len - 1), vocab_size)
        
        rewards = rewards[:, 1:].contiguous().view(-1)  # Ensure rewards match the size of labels

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(-1, self.num_emb)
        target = labels.view(-1)
        one_hot = F.one_hot(target, num_classes=self.num_emb).float()
        selected_log_prob = torch.sum(one_hot * log_probs, dim=-1)
        rewards = rewards.reshape(-1)
        logits_loss = -torch.sum(selected_log_prob * rewards) / (self.batch_size * self.sequence_length)

        # #Calculate classification loss
        # class_logits = class_logits.view(-1, self.num_classes)
        # class_labels = class_label.view(-1)
        # class_loss = F.cross_entropy(class_logits, class_labels)

        loss = logits_loss #+ class_loss

        return loss

    def train_step(self, x, class_label, rewards):
        """
        Performs a training step with class label.
        """
        self.optimizer.zero_grad()
        loss = self.train_loss(x, class_label, rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def generate(self, class_label, label_input=False): 
        """
        Generates a batch of samples along with their class labels.
        """
        hidden = self.init_hidden(self.batch_size)
        x = self.start_token.unsqueeze(1).to(self.emb.weight.device)  # [batch_size, 1]
        if self.use_cuda:
            hidden = tuple(h.cuda() for h in hidden)

        samples = []
        # is_end = torch.zeros(self.batch_size, dtype=torch.bool, device=x.device)  # Initialize is_end as False
        with torch.no_grad():
            for _ in range(self.sequence_length):
                if label_input:
                    logits, _, hidden = self.forward(x, class_label, hidden, label_input=True)
                else:
                    logits, _, hidden = self.forward(x, class_label, hidden)  # (batch_size, seq_len, vocab_size)
                probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)  # [batch_size, vocab_size]
                next_token = torch.multinomial(probs, 1).squeeze()  # [batch_size]
                samples.append(next_token)
                x = next_token.unsqueeze(1)  # [batch_size, 1]
        samples = torch.stack(samples, dim=1)
        first_pad_token = (samples == self.num_emb - 1).to(torch.long)
        first_pad_token = torch.argmax(first_pad_token, dim=-1)
        for i in range(first_pad_token.size(0)):
            if first_pad_token[i] < self.sequence_length - 1:
                samples[i, first_pad_token[i]+1:] = self.num_emb - 1
        return samples, class_label