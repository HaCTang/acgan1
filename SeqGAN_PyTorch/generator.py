# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Generator(nn.Module):
    """Generator """
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, num_classes, use_cuda,
                 sequence_length, start_token, learning_rate=0.001, reward_gamma=0.95, grad_clip=5.0):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.reward_gamma = reward_gamma
        self.grad_clip = grad_clip

        self.emb = nn.Embedding(num_emb, emb_dim)
        self.class_emb = nn.Embedding(num_classes, emb_dim) # Class embedding

        self.lstm = nn.LSTM(emb_dim * 2, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.adversarial_loss = F.cross_entropy
        self.auxiliary_loss = F.cross_entropy
        # self.init_params()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    # def init_params(self):
    #     for param in self.parameters():
    #         param.data.uniform_(-0.05, 0.05)
    
    def forward(self, x, class_label, hidden):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
            class_label: (batch_size, ), class label for the sequences
        """
        emb = self.emb(x) # (batch_size, seq_len, emb_dim)

        # Embedding for class labels and expand to match sequence length
        class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
        class_emb = class_emb.expand(-1, x.size(1), -1)       # (batch_size, seq_len, emb_dim)
        # Concatenate token embeddings with class embeddings
        combined_emb = torch.cat([emb, class_emb], dim=-1)  # (batch_size, seq_len, emb_dim * 2)

        output, hidden = self.lstm(combined_emb, hidden)
        pred = self.lin(output.contiguous().view(-1, self.hidden_dim))
        class_logits = self.classifier(output[:, -1, :]) 
        return pred.view(x.size(0), x.size(1), -1), class_logits, hidden
        #return pred

    def generate(self, class_label):
        """
        Generates a batch of samples along with their class labels.
        """
        # Initialize input and hidden states
        gen_x = torch.zeros(self.batch_size, self.sequence_length).long()
        hidden = self.init_hidden(self.batch_size)

        # Start token input
        x = torch.tensor([self.start_token] * self.batch_size).unsqueeze(1)  # [batch_size, 1]
        if self.use_cuda:
            x = x.cuda()
            gen_x = gen_x.cuda()

        # Generate sequence
        for i in range(self.sequence_length):
            logits, _, hidden = self.forward(x, class_label, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # [batch_size, num_emb]
            next_token = Categorical(probs).sample().unsqueeze(1)  # [batch_size, 1]
            gen_x[:, i] = next_token.squeeze()
            x = next_token

        return gen_x, class_label

    def pretrain_step(self, x, class_label):
        """
        Performs a pretraining step on the generator.
        Also known as supervised pretraining.
        """
        self.train()
        hidden = self.init_hidden(x.size(0))
        self.optimizer.zero_grad()

        # Forward pass
        logits, class_logits, _ = self.forward(x, class_label, hidden)
        logits = logits.view(-1, self.num_emb)  # [batch_size * seq_len, num_emb]
        target = x.view(-1)  # [batch_size * seq_len]

        # Calculate loss
        token_loss = self.adversarial_loss(logits, target)
        class_loss = self.auxiliary_loss(class_logits, class_label)
        loss = 0.5 * (token_loss + class_loss)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def train_step(self, x, class_label, rewards):
        """
        Performs a training step on the generator.
        Also known as unsupervised training.
        """
        self.train()
        hidden = self.init_hidden(x.size(0))
        self.optimizer.zero_grad()

        # Forward pass
        logits, _, _ = self.forward(x, class_label, hidden)
        logits = logits.view(-1, self.num_emb)  # [batch_size * seq_len, num_emb]
        target = x.view(-1)  # [batch_size * seq_len]

        # Calculate loss with rewards
        log_probs = F.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(target, self.num_emb).float()
        token_loss = -torch.sum(rewards.view(-1, 1) * one_hot * log_probs) / self.batch_size

        # Calculate class loss at the end of the sequence
        _, class_logits, _ = self.forward(x, class_label, hidden)
        class_loss = self.auxiliary_loss(class_logits, class_label)

        loss = 0.5 * (token_loss + class_loss)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

def test_generator():
    # Hyperparameters for testing
    num_emb = 10
    batch_size = 4
    emb_dim = 8
    hidden_dim = 16
    num_classes = 3
    use_cuda = False
    sequence_length = 5
    start_token = 0
    learning_rate = 0.001

    # Initialize generator
    generator = Generator(num_emb, batch_size, emb_dim, hidden_dim, num_classes, use_cuda, 
                          sequence_length, start_token, learning_rate)

    # Generate sample class labels
    class_labels = torch.randint(0, num_classes, (batch_size,))

    # Test generate function
    generated_seq, generated_labels = generator.generate(class_labels)
    print("Generated Sequence:", generated_seq)
    print("Generated Labels:", generated_labels)

    # Create random input sequence and class labels for pretrain step
    input_seq = torch.randint(0, num_emb, (batch_size, sequence_length))
    pretrain_loss = generator.pretrain_step(input_seq, class_labels)
    print("Pretrain Loss:", pretrain_loss)

    # Create random rewards for train step
    rewards = torch.rand(batch_size, sequence_length)
    train_loss = generator.train_step(input_seq, class_labels, rewards)
    print("Train Loss:", train_loss)

if __name__ == "__main__":
    test_generator()

    # def step(self, x, class_label, h, c):
    #     """
    #     Args:
    #         x: (batch_size,  1), sequence of tokens generated by generator
    #         class_label: (batch_size, ), class label for the sequences
    #         h: (1, batch_size, hidden_dim), lstm hidden state
    #         c: (1, batch_size, hidden_dim), lstm cell state
    #     """
    #     emb = self.emb(x)

    #     # Class embedding
    #     class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
        
    #     # Concatenate token embedding with class embedding
    #     combined_emb = torch.cat([emb, class_emb], dim=-1)  # (batch_size, 1, emb_dim * 2)
  
    #     output, (h, c) = self.lstm(combined_emb, (h, c))
    #     pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
    #     return pred, h, c

    # def sample(self, batch_size, seq_len, class_label, x=None):
    #     res = []
    #     flag = False # whether sample from zero
    #     if x is None:
    #         flag = True
    #     if flag:
    #         x = Variable(torch.zeros((batch_size, 1)).long())
    #     if self.use_cuda:
    #         x = x.cuda()
    #     h, c = self.init_hidden(batch_size)
    #     samples = []
    #     if flag:
    #         for i in range(seq_len):
    #             output, h, c = self.step(x, class_label, h, c)
    #             x = output.multinomial(1)
    #             samples.append(x)
    #     else:
    #         given_len = x.size(1)
    #         lis = x.chunk(x.size(1), dim=1)
    #         for i in range(given_len):
    #             output, h, c = self.step(lis[i], class_label, h, c)
    #             samples.append(lis[i])
    #         x = output.multinomial(1)
    #         for i in range(given_len, seq_len):
    #             samples.append(x)
    #             output, h, c = self.step(x, class_label, h, c)
    #             x = output.multinomial(1)
    #     output = torch.cat(samples, dim=1)
    #     return output
