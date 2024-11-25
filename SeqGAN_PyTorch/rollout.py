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
    def __init__(self, generator, update_rate, pad_num):
        super(Rollout, self).__init__()
        '''
        lstm: LSTM model is Generator or WGenerator
        '''
        # Set parameters and define model architecture
        generator = copy.deepcopy(generator)
        self.generator = generator
        self.update_rate = update_rate
        self.pad_num = pad_num

        self.vocab_size = generator.vocab_size
        self.batch_size = generator.batch_size
        self.seq_emb_dim = generator.seq_emb_dim
        self.class_emb_dim = generator.class_emb_dim
        self.hidden_dim = generator.hidden_dim
        self.use_cuda = generator.use_cuda
        self.sequence_length = generator.sequence_length
        self.start_token = generator.start_token

        self.seq_emb = generator.seq_emb
        self.class_emb = generator.class_emb
        self.lstm = generator.seq_lstm # nn.LSTM(seq_emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lin = generator.lin
        

    def forward(self, x, class_label, given_num):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
            given_num: (int) - Number of tokens to consider in the input. 
                               When current index i < given_num, use the provided tokens 
                               as the input at each time step
                               When current index i >= given_num, start roll-out, use the 
                               output as time step t as the input at time step t+1
        """
        batch_size, seq_len = x.size()
        outputs = []
        device = x.device
        h, c = (torch.zeros(2, batch_size, self.hidden_dim, device=device),
                torch.zeros(2, batch_size, self.hidden_dim, device=device))
        seq_emb = self.seq_emb(x) # (batch_size, seq_len, seq_emb_dim)

        for i in range(given_num):
            token_emb = seq_emb[:, i, :].unsqueeze(1) # (batch_size, 1, seq_emb_dim)
            _, (h, c) = self.lstm(token_emb, (h, c))
            outputs.append(x[:, i])
        token_emb = seq_emb[:, given_num-1, :].unsqueeze(1)
        for i in range(given_num, seq_len):
            seq_output, (h, c) = self.lstm(token_emb, (h, c))
            logits = self.lin(seq_output.contiguous().view(-1, self.hidden_dim)) # (batch_size, vocab_size)
            logits = logits.view(batch_size, -1, self.vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze()
            # next_token = torch.argmax(probs, dim=-1)
            outputs.append(next_token)
            token_emb = self.seq_emb(next_token).unsqueeze(1) # (batch_size, 1, seq_emb_dim)

        outputs = torch.stack(outputs, dim=1)
        # first_pad_token = (outputs == self.pad_num).to(torch.long)
        # first_pad_token = torch.argmax(first_pad_token, dim=-1)
        # for i in range(first_pad_token.size(0)):
        #     if first_pad_token[i] < self.sequence_length - 1:
        #         outputs[i, first_pad_token[i]+1:] = self.pad_num
        return outputs

    def get_reward(self, input_x, class_label, rollout_num, dis, reward_fn=None, Dweight1=1.0):
        """
        Calculates the rewards for a list of SMILES strings.
        Args:
            input_x: (batch_size, seq_len), sequence of tokens generated by generator
            rollout_num: (int) - Number of rollouts to perform
            dis: Discriminator model
            reward_fn: (function) - Reward function to use
            Dweight1: (float) - Weight to assign to the discriminator output
                                (1-Dweight1) is the weight assigned to the reward function output
        """
        reward_weight = 1 - Dweight1
        rewards = np.zeros((input_x.size(0), self.sequence_length))
        
        for i in range(rollout_num):
            already = []
            for given_num in range(1, self.sequence_length):
                # 生成rollout序列
                generated_seqs = self.forward(input_x, class_label, given_num).detach()
                gind = np.array(range(len(generated_seqs)))
                
                # 判别器评分
                dis_output, _ = dis(generated_seqs.to(dis.emb.weight.device))
                # 使用softmax获取概率
                dis_output = F.softmax(dis_output, dim=1)
                dis_rewards = dis_output[:, 1].detach().cpu().numpy()  # 取第二维作为真实样本的概率
                
                if reward_fn:
                    dis_rewards = Dweight1 * dis_rewards
                    generated_seqs_np = generated_seqs.cpu().numpy()
                    
                    # 处理已完成的序列
                    for k, r in reversed(already):
                        generated_seqs_np = np.delete(generated_seqs_np, k, 0)
                        gind = np.delete(gind, k, 0)
                        dis_rewards[k] += reward_weight * r
                    
                    # 计算reward_fn奖励
                    if generated_seqs_np.size:
                        fn_rewards = reward_fn(generated_seqs_np)
                        
                    # 添加reward_fn奖励    
                    for k, r in zip(gind, fn_rewards):
                        dis_rewards[k] += reward_weight * r
                    
                    # 记录本轮完成的序列
                    for j, k in enumerate(gind):
                        if input_x[k, given_num] == self.pad_num and input_x[k, given_num-1] == self.pad_num:
                            already.append((k, fn_rewards[j]))
                    already = sorted(already, key=lambda el: el[0])
                
                rewards[:, given_num-1] += dis_rewards
            
            # 计算最后一个时间步的奖励
            dis_output, _ = dis(input_x.to(dis.emb.weight.device))
            dis_output = F.softmax(dis_output, dim=1)
            dis_rewards = dis_output[:, 1].detach().cpu().numpy()
            
            if reward_fn:
                input_x_np = input_x.cpu().numpy()
                dis_rewards = Dweight1 * dis_rewards
                fn_rewards = reward_fn(input_x_np)
                dis_rewards += reward_weight * fn_rewards
                
            rewards[:, -1] += dis_rewards
        
        # 平均每个rollout的奖励
        rewards = rewards / rollout_num
        return rewards


    def update_params(self, original_generator):
        """
        Updates all parameters in the rollout's LSTM. 
        Use update_rate to update the parameters 
        along with the generator's LSTM.
        """
        # with torch.no_grad():
        #     for param, lstm_param in zip(self.parameters(), self.generator.parameters()):
        #         param.data = self.update_rate * param.data + (1 - self.update_rate) * lstm_param.data'
        # import pdb
        # pdb.set_trace()
        for name, param in self.named_parameters():
            if 'lstm' in name:
                # print(name, '.'.join(name.split('.')[1:]))
                param.data = self.update_rate * param.data + (1 - self.update_rate) * original_generator.state_dict()['.'.join(name.split('.')[1:])].data
            elif 'lin' in name:
                # print(name, '.'.join(name.split('.')[1:]))
                param.data = self.update_rate * param.data + (1 - self.update_rate) * original_generator.state_dict()['.'.join(name.split('.')[1:])].data


# class Rollout(nn.Module):
#     """
#     Class for the rollout policy model.
#     """
#     def __init__(self, lstm, update_rate, pad_num):
#         super(Rollout, self).__init__()
#         '''
#         lstm: LSTM model is Generator or WGenerator
#         '''
#         # Set parameters and define model architecture
#         self.lstm = lstm

#         self.update_rate = update_rate
#         self.pad_num = pad_num

#         self.vocab_size = lstm.vocab_size
#         self.batch_size = lstm.batch_size
#         self.seq_emb_dim = lstm.seq_emb_dim
#         self.class_emb_dim = lstm.class_emb_dim
#         self.hidden_dim = lstm.hidden_dim
#         self.use_cuda = lstm.use_cuda
#         self.sequence_length = lstm.sequence_length
#         self.start_token = lstm.start_token
#         self.learning_rate = lstm.learning_rate

#         self.seq_emb = lstm.seq_emb
#         self.class_emb = lstm.class_emb

#         self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
#         self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

#     def forward(self, x, class_label, given_num):
#         # Sequence of tokens generated by generator
#         x = x.to(self.seq_emb.weight.device)
#         emb = self.seq_emb(x).view(self.batch_size, self.sequence_length, self.seq_emb_dim)

#         # # Embedding for class labels and expand to match sequence length
#         # class_label = class_label.to(self.class_emb.weight.device)
#         # class_emb = self.class_emb(class_label).unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch_size, 1, emb_dim)
#         # class_emb = class_emb.expand(-1, self.sequence_length, -1)       # (batch_size, seq_len, emb_dim)
        
#         # # Concatenate token embeddings with class embeddings
#         # self.processed_x = torch.cat([emb, class_emb], dim=-1).to(emb.device)  # (seq_len, batch_size, emb_dim+class_emb_dim)

#         gen_x = []
#         h_tm1 = (torch.zeros(self.batch_size, self.hidden_dim).to(x.device),
#                  torch.zeros(self.batch_size, self.hidden_dim).to(x.device))
        
#         # When current index i < given_num, use the provided tokens as the input at each time step
#         for i in range(given_num):
#             # x_t = self.processed_x[:, i, :]
#             x_t = emb[:, i, :]
#             h_tm1 = self.g_recurrent_unit(x_t, h_tm1)
#             gen_x.append(x[:, i])


# #to do: devide class roll-out and sample roll-out
    
#         # When current index i >= given_num, start roll-out, use the output at time step t as the input at time step t+1
#         for i in range(given_num, self.sequence_length):
#             if i == given_num:
#                 x_t = emb[:, i, :]
#             else:
#                 next_token = next_token.unsqueeze(1)  # Ensure next_token has shape [batch_size, 1]
#                 token_emb = self.seq_emb(next_token).squeeze(1)  # [batch_size, emb_dim]
#                 x_t = token_emb
#             # print("22:", x_t.shape)
#             h_tm1 = self.g_recurrent_unit(x_t, h_tm1)
#             o_t = self.g_output_unit(h_tm1[0])  # logits not prob
#             log_prob = F.log_softmax(o_t, dim=-1)
#             next_token = torch.multinomial(log_prob.exp(), 1).squeeze()
#             gen_x.append(next_token)

#         return torch.stack(gen_x, dim=1)  # batch_size x seq_length

#     def get_reward(self, input_x, class_label, rollout_num, dis, reward_fn=None, Dweight1=1, Dweight2=1):
#         """Calculates the rewards for a list of SMILES strings."""
#         reward_weight1 = 1 - Dweight1
#         rewards = [0] * (self.sequence_length)
#         for _ in range(rollout_num):
#             already = []
#             for given_num in range(1, self.sequence_length):
#                 generated_seqs = self.forward(input_x, class_label, given_num).detach().to(dis.emb.weight.device)
#                 gind = np.array(range(len(generated_seqs)))
                
#                 ypred_for_auc, _ = dis(generated_seqs.to(dis.emb.weight.device))
#                 ypred = ypred_for_auc.detach().clone()
                
#                 if reward_fn:
#                     ypred = Dweight1 * ypred
#                     rew = reward_fn(generated_seqs.cpu().numpy())
                    
#                     for k, r in zip(gind, rew):
#                         ypred[k] += reward_weight1 * r

#                     for j, k in enumerate(gind):
#                         if input_x[k, given_num] == self.pad_num and input_x[k, given_num - 1] == self.pad_num:
#                             already.append((k, rew[j]))

#                     already = sorted(already, key=lambda el: el[0])

#                 if len(rewards) == 0:
#                     rewards.append(ypred)
#                 else:
#                     rewards[given_num - 1] += ypred
            
#             ypred_for_auc, _ = dis(input_x.to(dis.emb.weight.device))
#             ypred_for_auc = ypred_for_auc.detach()
            
#             if reward_fn:
#                 input_x_list = input_x.cpu().tolist()
#                 ypred = Dweight1 * ypred_for_auc + reward_weight1 * torch.tensor(reward_fn(input_x_list).reshape(-1, 1), device=ypred_for_auc.device, dtype=ypred_for_auc.dtype)
#             else:
#                 ypred = ypred_for_auc
            
#             if len(rewards) == 0:
#                 rewards.append(ypred)
#             else:
#                 rewards[-1] += ypred
        
#         rewards = np.transpose(np.array([reward.cpu().numpy() for reward in rewards])) / (1.0 * rollout_num)  # batch_size x seq_length
#         flattened_rewards = rewards[0]
#         return flattened_rewards

#     def create_recurrent_unit(self):
#         """Defines the recurrent process in the LSTM."""
#         return nn.LSTMCell(self.seq_emb_dim, self.hidden_dim)

#     def create_output_unit(self):
#         """Defines the output process in the LSTM."""
#         return nn.Linear(self.hidden_dim, self.vocab_size)

#     def update_params(self):
#         """Updates all parameters in the rollout's LSTM."""
#         with torch.no_grad():
#             for param, lstm_param in zip(self.parameters(), self.lstm.parameters()):
#                 param.data = self.update_rate * param.data + (1 - self.update_rate) * lstm_param.data

