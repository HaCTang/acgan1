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
    """
    Class for the rollout policy model.
    """
    def __init__(self, lstm, update_rate, pad_num):
        super(Rollout, self).__init__()
        '''
        lstm: LSTM model is Generator or WGenerator
        '''
        # Set parameters and define model architecture
        self.lstm = lstm

        self.update_rate = update_rate
        self.pad_num = pad_num

        self.num_emb = lstm.num_emb
        self.batch_size = lstm.batch_size
        self.emb_dim = lstm.emb_dim
        self.class_emb_dim = lstm.class_emb_dim
        self.hidden_dim = lstm.hidden_dim
        self.use_cuda = lstm.use_cuda
        self.sequence_length = lstm.sequence_length
        self.start_token = lstm.start_token
        self.learning_rate = lstm.learning_rate

        self.emb = lstm.emb
        self.class_emb = self.lstm.class_emb

        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

    def forward(self, x, class_label, given_num):
        # Sequence of tokens generated by generator
        x = x.to(self.g_embeddings.weight.device)
        emb = self.g_embeddings(x).view(self.batch_size, self.sequence_length, self.emb_dim)

        # Embedding for class labels and expand to match sequence length
        class_label = class_label.to(self.class_emb.weight.device)
        class_emb = self.class_emb(class_label).unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch_size, 1, emb_dim)
        class_emb = class_emb.expand(-1, self.sequence_length, -1)       # (batch_size, seq_len, emb_dim)
        
        # Concatenate token embeddings with class embeddings
        self.processed_x = torch.cat([emb, class_emb], dim=-1).to(emb.device)  # (seq_len, batch_size, emb_dim+class_emb_dim)

        gen_x = []
        h_tm1 = (torch.zeros(self.batch_size, self.hidden_dim).to(x.device),
                 torch.zeros(self.batch_size, self.hidden_dim).to(x.device))
        
        # When current index i < given_num, use the provided tokens as the input at each time step
        for i in range(given_num):
            x_t = self.processed_x[:, i, :]
            h_tm1 = self.g_recurrent_unit(x_t, h_tm1)
            gen_x.append(x[:, i])


#to do: devide class roll-out and sample roll-out
    
        # When current index i >= given_num, start roll-out, use the output at time step t as the input at time step t+1
        for i in range(given_num, self.sequence_length):
            if i == given_num:
                x_t = self.processed_x[:, i, :]
            else:
                next_token = next_token.unsqueeze(1)  # Ensure next_token has shape [batch_size, 1]
                token_emb = self.g_embeddings(next_token).squeeze(1)  # [batch_size, emb_dim]
                class_emb_current = class_emb[:, i, :]  # [batch_size, emb_dim]
                x_t = torch.cat([token_emb, class_emb_current], dim=-1)  # [batch_size, emb_dim+class_emb_dim]
            # print("22:", x_t.shape)
            h_tm1 = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_tm1[0])  # logits not prob
            log_prob = F.log_softmax(o_t, dim=-1)
            next_token = torch.multinomial(log_prob.exp(), 1).squeeze()
            gen_x.append(next_token)

        return torch.stack(gen_x, dim=1)  # batch_size x seq_length

    def get_reward(self, input_x, class_label, rollout_num, dis, reward_fn=None, D_weight=1):
        """Calculates the rewards for a list of SMILES strings."""
        reward_weight = 1 - D_weight
        rewards = [0] * (self.sequence_length)
        for _ in range(rollout_num):
            already = []
            for given_num in range(1, self.sequence_length):
                generated_seqs = self.forward(input_x, class_label, given_num).detach().to(dis.emb.weight.device)
                gind = np.array(range(len(generated_seqs)))
                
                dis_output = dis(generated_seqs.to(dis.emb.weight.device))
                if isinstance(dis_output, tuple):
                    ypred_for_auc, yclasspred_for_auc = dis_output
                else:
                    ypred_for_auc = dis_output
                    yclasspred_for_auc = None

                if yclasspred_for_auc is not None:
                    yclasspred_for_auc = yclasspred_for_auc.detach()
                ypred_for_auc = ypred_for_auc.detach()
                
                ypred = ypred_for_auc.clone()
                yclasspred = yclasspred_for_auc.clone() if yclasspred_for_auc is not None else None
                
                if reward_fn:
                    ypred = D_weight * ypred
                    rew = reward_fn(generated_seqs.cpu().numpy())
                    
                    for k, r in zip(gind, rew):
                        ypred[k] += reward_weight * r

                    for j, k in enumerate(gind):
                        if input_x[k, given_num] == self.pad_num and input_x[k, given_num - 1] == self.pad_num:
                            already.append((k, rew[j]))

                    already = sorted(already, key=lambda el: el[0])

                if len(rewards) == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            
            dis_output = dis(input_x.to(dis.emb.weight.device))
            if isinstance(dis_output, tuple):
                ypred_for_auc, yclasspred_for_auc = dis_output
            else:
                ypred_for_auc = dis_output
                yclasspred_for_auc = None
            yclasspred_for_auc = yclasspred_for_auc.detach() if yclasspred_for_auc is not None else None
            ypred_for_auc = ypred_for_auc.detach()
            
            if reward_fn:
                input_x_list = input_x.cpu().tolist()
                ypred = D_weight * ypred_for_auc + reward_weight * torch.tensor(reward_fn(input_x_list).reshape(-1, 1), device=ypred_for_auc.device, dtype=ypred_for_auc.dtype)
            else:
                ypred = ypred_for_auc
            
            if len(rewards) == 0:
                rewards.append(ypred)
            else:
                rewards[-1] += ypred
        
        rewards = np.transpose(np.array([reward.cpu().numpy() for reward in rewards])) / (1.0 * rollout_num)  # batch_size x seq_length
        flattened_rewards = rewards[0]
        return flattened_rewards

    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM."""
        return nn.LSTMCell(self.emb_dim + self.class_emb_dim, self.hidden_dim)

    def create_output_unit(self):
        """Defines the output process in the LSTM."""
        return nn.Linear(self.hidden_dim, self.num_emb)

    def update_params(self):
        """Updates all parameters in the rollout's LSTM."""
        with torch.no_grad():
            self.g_embeddings = nn.Embedding.from_pretrained(self.lstm.emb.weight.clone(), freeze=False).to(self.lstm.emb.weight.device)
            self.class_emb = nn.Embedding.from_pretrained(self.lstm.class_emb.weight.clone(), freeze=False).to(self.lstm.class_emb.weight.device)
            lstm_state_dict = self.lstm.lstm.state_dict()
            lstm_cell_state_dict = {
                'weight_ih': lstm_state_dict['weight_ih_l0'],
                'weight_hh': lstm_state_dict['weight_hh_l0'],
                'bias_ih': lstm_state_dict['bias_ih_l0'],
                'bias_hh': lstm_state_dict['bias_hh_l0']
            }
            self.g_recurrent_unit.load_state_dict(lstm_cell_state_dict)
            self.g_output_unit.load_state_dict(self.lstm.lin.state_dict())