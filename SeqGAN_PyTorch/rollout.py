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

class Rollout(nn.Module):
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

        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

    def forward(self, x, class_label, given_num):
        """Forward pass for the rollout policy."""

        device = next(self.parameters()).device
        x = x.to(device)
        class_label = class_label.to(device)

        # processed for batch
        emb_x = self.emb(x)  # batch_size x seq_length x emb_dim
        emb_class = self.class_emb(class_label).unsqueeze(1)  # batch_size x 1 x emb_dim
        emb_class = emb_class.expand(-1, x.size(1), -1)  # batch_size x seq_length x emb_dim
        inputs = torch.cat([emb_x, emb_class], dim=-1)  # batch_size x seq_length x (2*emb_dim)
        processed_x = inputs.permute(1, 0, 2)  # seq_length x batch_size x (2*emb_dim)

        h_0 = torch.zeros(2, self.batch_size, self.hidden_dim, device=device)

        gen_x = torch.zeros(self.sequence_length, self.batch_size, dtype=torch.int64, device=device)

        # When current index i < given_num, use the provided tokens as the input at each time step
        for i in range(given_num):
            h_0 = self.g_recurrent_unit(processed_x[i], h_0)
            gen_x[i] = x[:, i]

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        for i in range(given_num, self.sequence_length):
            h_0 = self.g_recurrent_unit(processed_x[i], h_0)
            o_t = self.g_output_unit(h_0)  # batch x vocab, logits not prob
            log_prob = F.log_softmax(o_t, dim=-1)
            next_token = torch.multinomial(torch.exp(log_prob), 1).squeeze()
            next_token_emb = self.emb(next_token)  # batch_size x emb_dim
            processed_x[i] = torch.cat([next_token_emb, emb_class[:, i, :]], dim=-1)  # update with new token embedding
            gen_x[i] = next_token

        gen_x = gen_x.permute(1, 0)  # batch_size x seq_length
        return gen_x, class_label

    def get_reward(self, input_x, rollout_num, dis, reward_fn=None, D_weight=1):
        """Calculates the rewards for a list of SMILES strings."""

        reward_weight = 1 - D_weight
        rewards = []
        device = next(self.parameters()).device
        input_x = input_x.to(device)

        for i in range(rollout_num):

            already = []
            for given_num in range(1, self.sequence_length):
                generated_seqs, _ = self.forward(input_x, torch.zeros(input_x.size(0), dtype=torch.long), given_num)
                ypred_for_auc = dis(generated_seqs).detach().cpu().numpy()
                ypred = np.array([item[1] for item in ypred_for_auc])

                if reward_fn:

                    ypred = D_weight * ypred
                    # Delete sequences that are already finished,
                    # and add their rewards
                    for k, r in reversed(already):
                        generated_seqs = np.delete(generated_seqs, k, 0)
                        ypred[k] += reward_weight * r

                    # If there are still seqs, calculate rewards
                    if generated_seqs.size:
                        rew = reward_fn(generated_seqs)

                    # Add the just calculated rewards
                    for k, r in zip(range(len(generated_seqs)), rew):
                        ypred[k] += reward_weight * r

                    # Choose the seqs finished in the last iteration
                    for j, k in enumerate(range(len(generated_seqs))):
                        if input_x[k, given_num] == self.pad_num and input_x[k, given_num-1] == self.pad_num:
                            already.append((k, rew[j]))
                    already = sorted(already, key=lambda el: el[0])

                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # Last char reward
            ypred_for_auc = dis(input_x).detach().cpu().numpy()
            if reward_fn:
                ypred = D_weight * np.array([item[1] for item in ypred_for_auc])
                ypred += reward_weight * reward_fn(input_x.cpu().numpy())
            else:
                ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[-1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM."""

        self.Wi = self.lstm.Wi
        self.Ui = self.lstm.Ui
        self.bi = self.lstm.bi

        self.Wf = self.lstm.Wf
        self.Uf = self.lstm.Uf
        self.bf = self.lstm.bf

        self.Wog = self.lstm.Wog
        self.Uog = self.lstm.Uog
        self.bog = self.lstm.bog

        self.Wc = self.lstm.Wc
        self.Uc = self.lstm.Uc
        self.bc = self.lstm.bc

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = hidden_memory_tm1[0], hidden_memory_tm1[1]

            # Input Gate
            i = torch.sigmoid(torch.matmul(x, self.Wi) + torch.matmul(previous_hidden_state, self.Ui) + self.bi)

            # Forget Gate
            f = torch.sigmoid(torch.matmul(x, self.Wf) + torch.matmul(previous_hidden_state, self.Uf) + self.bf)

            # Output Gate
            o = torch.sigmoid(torch.matmul(x, self.Wog) + torch.matmul(previous_hidden_state, self.Uog) + self.bog)

            # New Memory Cell
            c_ = torch.tanh(torch.matmul(x, self.Wc) + torch.matmul(previous_hidden_state, self.Uc) + self.bc)

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * torch.tanh(c)

            return torch.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        """Defines the output process in the LSTM."""

        self.Wo = self.lstm.Wo
        self.bo = self.lstm.bo

        def unit(hidden_memory_tuple):
            hidden_state, _ = hidden_memory_tuple
            # hidden_state : batch x hidden_dim
            logits = torch.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit

    def update_params(self):
        """Updates all parameters in the rollout's LSTM."""
        self.g_embeddings = self.lstm.g_embeddings
        self.g_recurrent_unit = self.create_recurrent_unit()
        self.g_output_unit = self.create_output_unit()


if __name__ == "__main__":
    # Sample LSTM model for testing
    class SampleLSTM(nn.Module):
        def __init__(self):
            super(SampleLSTM, self).__init__()
            self.num_emb = 10
            self.batch_size = 4
            self.emb_dim = 8
            self.hidden_dim = 16
            self.sequence_length = 6
            self.start_token = 0
            self.learning_rate = 0.01
            self.use_cuda = False

            self.emb = nn.Embedding(self.num_emb, self.emb_dim)
            self.class_emb = nn.Embedding(3, self.emb_dim)

            self.Wi = nn.Parameter(torch.randn(2 * self.emb_dim, self.hidden_dim))
            self.Ui = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
            self.bi = nn.Parameter(torch.zeros(self.hidden_dim))

            self.Wf = nn.Parameter(torch.randn(2 * self.emb_dim, self.hidden_dim))
            self.Uf = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
            self.bf = nn.Parameter(torch.zeros(self.hidden_dim))

            self.Wog = nn.Parameter(torch.randn(2 * self.emb_dim, self.hidden_dim))
            self.Uog = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
            self.bog = nn.Parameter(torch.zeros(self.hidden_dim))

            self.Wc = nn.Parameter(torch.randn(2 * self.emb_dim, self.hidden_dim))
            self.Uc = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
            self.bc = nn.Parameter(torch.zeros(self.hidden_dim))

            self.Wo = nn.Parameter(torch.randn(self.hidden_dim, self.num_emb))
            self.bo = nn.Parameter(torch.zeros(self.num_emb))

        def forward(self, x):
            pass

    lstm = SampleLSTM()
    rollout = Rollout(lstm=lstm, update_rate=0.8, pad_num=0)

    # Create dummy input
    dummy_input = torch.randint(0, lstm.num_emb, (lstm.batch_size, lstm.sequence_length), dtype=torch.long)
    dummy_class_input = torch.randint(0, 3, (lstm.batch_size,))
    given_num = torch.tensor(3)

    # Run forward pass
    generated_output = rollout(dummy_input, dummy_class_input, given_num)
    print("Generated Output:", generated_output)

    # def get_reward(self, x, num, discriminator, class_labels):
    #     """
    #     Args:
    #         x : (batch_size, seq_len) input data (generated sequences)
    #         num : roll-out number
    #         discriminator : discrimanator model
    #         class_labels: (batch_size,) ground truth class labels
    #     """
    #     rewards = []
    #     batch_size = x.size(0)
    #     seq_len = x.size(1)
    #     for i in range(num):
    #         for l in range(1, seq_len):
    #             data = x[:, 0:l]
    #             samples = self.own_model.sample(batch_size, seq_len, data)
                
    #             # Get both real/fake prediction and class prediction from discriminator
    #             real_fake_pred, class_pred = discriminator(samples)
    #             real_fake_pred = real_fake_pred.cpu().data.numpy()
    #             class_pred = class_pred.cpu().data.numpy()
                
    #             # Compute rewards based on real/fake prediction
    #             real_fake_reward = real_fake_pred[:, 0]  # Assuming 0 index for real
    #             # Optionally, include class prediction accuracy in reward
    #             class_reward = np.sum(np.argmax(class_pred, axis=1) == class_labels.cpu().numpy()) / float(batch_size)
    #             total_reward = real_fake_reward + class_reward  # Combine both rewards
    #             if i == 0:
    #                 rewards.append(total_reward)
    #             else:
    #                 rewards[l-1] += total_reward

    #         # for the last token
    #         real_fake_pred, class_pred = discriminator(x)
    #         real_fake_pred = real_fake_pred.cpu().data.numpy()
    #         class_pred = class_pred.cpu().data.numpy()
            
    #         real_fake_reward = real_fake_pred[:, 0]  # Real/fake reward
    #         class_reward = np.sum(np.argmax(class_pred, axis=1) == class_labels.cpu().numpy()) / float(batch_size)
            
    #         total_reward = real_fake_reward + class_reward  # Combine rewards for the last token
    
    #         if i == 0:
    #             rewards.append(total_reward)
    #         else:
    #             rewards[seq_len-1] += total_reward
    #     rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
    #     return rewards

    # def update_params(self):
    #     dic = {}
    #     for name, param in self.ori_model.named_parameters():
    #         dic[name] = param.data
    #     for name, param in self.own_model.named_parameters():
    #         if name.startswith('emb'):
    #             param.data = dic[name]
    #         else:
    #             param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]