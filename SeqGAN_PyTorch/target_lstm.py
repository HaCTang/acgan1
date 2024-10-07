# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class TargetLSTM(nn.Module):
    """Target Lstm """
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, 
                 sequence_length, start_token, num_classes, use_cuda):
        super(TargetLSTM, self).__init__()
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.temperature = 1.0

        torch.manual_seed(66)
        
        # Token and class embedding layers
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(emb_dim * 2, hidden_dim, batch_first=True)  # Concatenate token and class embeddings
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.classifier = nn.Linear(hidden_dim, num_classes) # Classifier for class label prediction
        
        # self.init_params()

    def forward(self, x, class_label, hidden):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
            class_label: (batch_size,), class label for conditional generation
        """
        emb = self.emb(x)
        
        # Get class embedding and expand to match sequence length
        class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
        class_emb = class_emb.expand(-1, x.size(1), -1)       # (batch_size, seq_len, emb_dim)
        # Concatenate token embeddings with class embeddings
        combined_emb = torch.cat([emb, class_emb], dim=-1)  # (batch_size, seq_len, emb_dim * 2)
        
        output, hidden = self.lstm(combined_emb, hidden)  # Pass through LSTM
        
        pred = self.lin(output.contiguous().view(-1, self.hidden_dim))
        class_logits = self.classifier(output[:, -1, :])  # Predict class label from the last output
        return pred.view(x.size(0), x.size(1), -1), class_logits, hidden

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim))).to('cuda' if self.use_cuda else 'cpu')
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim))).to('cuda' if self.use_cuda else 'cpu')
        return h, c

    # def init_params(self):
    #     for param in self.parameters():
    #         param.data.normal_(0, 1)
    
    def generate(self, class_label):
        gen_x = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long).to('cuda' if self.use_cuda else 'cpu')
        hidden = self.init_hidden(self.batch_size)

        x_t = torch.full((self.batch_size,), self.start_token, dtype=torch.long).to('cuda' if self.use_cuda else 'cpu')
        for i in range(self.sequence_length):
            x_t_embedded = self.emb(x_t).unsqueeze(1)  # [batch_size, 1, emb_dim]
            class_emb = self.class_emb(class_label).unsqueeze(1)  # [batch_size, 1, emb_dim]
            combined_emb = torch.cat([x_t_embedded, class_emb], dim=-1)  # [batch_size, 1, emb_dim * 2]
            output, hidden = self.lstm(combined_emb, hidden)  # [batch_size, 1, hidden_dim]
            logits = self.lin(output.squeeze(1))  # [batch_size, num_emb]
            prob = F.softmax(logits / self.temperature, dim=-1)  # [batch_size, num_emb]
            x_t = torch.multinomial(prob, 1).squeeze(1)  # [batch_size]
            gen_x[:, i] = x_t

        return gen_x, class_label

    def pretrain_loss(self, x, class_label):
        logits, class_logits, _ = self.forward(x, class_label, self.init_hidden(x.size(0)))  # [batch_size, sequence_length, num_emb]
        logits = logits.view(-1, self.num_emb)  # [batch_size * sequence_length, num_emb]
        targets = x.view(-1)  # [batch_size * sequence_length]
        token_loss = F.cross_entropy(logits, targets)

        # Class label prediction loss
        class_loss = F.cross_entropy(class_logits, class_label)

        # Total loss
        loss = token_loss + class_loss
        return loss

# Example usage
if __name__ == "__main__":
    num_emb = 5000
    batch_size = 64
    emb_dim = 32
    hidden_dim = 64
    sequence_length = 20
    start_token = 0
    num_classes = 10
    use_cuda = torch.cuda.is_available()

    model = TargetLSTM(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, num_classes, use_cuda).to('cuda' if use_cuda else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dummy data for pretraining
    x = torch.randint(0, num_emb, (batch_size, sequence_length), dtype=torch.long).to('cuda' if use_cuda else 'cpu')
    class_label = torch.randint(0, num_classes, (batch_size,), dtype=torch.long).to('cuda' if use_cuda else 'cpu')
    pretrain_loss = model.pretrain_loss(x, class_label)

    optimizer.zero_grad()
    pretrain_loss.backward()
    optimizer.step()

    # Generate sequences
    generated_sequences = model.generate(class_label)
    print(generated_sequences)


    # def step(self, x, class_label, h, c):
    #     """
    #     Args:
    #         x: (batch_size, 1), sequence of tokens generated by generator
    #         class_label: (batch_size,), class label for conditional generation
    #         h: (1, batch_size, hidden_dim), lstm hidden state
    #         c: (1, batch_size, hidden_dim), lstm cell state
    #     """
    #     emb = self.emb(x)  # (batch_size, 1, emb_dim)
        
    #     # Class embedding
    #     class_emb = self.class_emb(class_label).unsqueeze(1)  # (batch_size, 1, emb_dim)
        
    #     # Concatenate token embedding with class embedding
    #     combined_emb = torch.cat([emb, class_emb], dim=-1)  # (batch_size, 1, emb_dim * 2)
        
    #     output, (h, c) = self.lstm(combined_emb, (h, c))
    #     pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
    #     return pred, h, c

    # def sample(self, batch_size, seq_len, class_label):
    #     """
    #     Sample sequences conditionally based on class_label.
        
    #     Args:
    #         batch_size: Number of sequences to generate
    #         seq_len: Length of sequences
    #         class_label: (batch_size,), class labels for conditional generation
    #     """
    #     with torch.no_grad():
    #         x = Variable(torch.zeros((batch_size, 1)).long())
    #         if self.use_cuda:
    #             x = x.cuda()
    #         h, c = self.init_hidden(batch_size)
    #         samples = []
    #         for i in range(seq_len):
    #             output, h, c = self.step(x, class_label, h, c)
    #             x = output.multinomial(1)
    #             samples.append(x)
    #         output = torch.cat(samples, dim=1)
    #         return output
    #     return None
