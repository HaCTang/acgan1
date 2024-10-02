from __future__ import absolute_import, division, print_function
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from builtins import range
from collections import OrderedDict, defaultdict
import numpy as np
import random
import dill as pickle

from SeqGAN_PyTorch.data_iter import GenDataIter, DisDataIter
from SeqGAN_PyTorch.generator import Generator
from SeqGAN_PyTorch.discriminator import Discriminator
from SeqGAN_PyTorch.rollout import Rollout

from rdkit import rdBASE
import pandas as pd
from tqdm import tqdm, trange
import SeqGAN_PyTorch.mol_metrics

import math
import argparse

class ACSeqGAN(object):
    """Main class, where every interaction between the user
    and the backend is performed.
    """

    def __init__(self, name, metrics_module, params={},
                 verbose=True):
        """Parameter initialization.

        Arguments
        -----------

            - name. String which will be used to identify the
            model in any folders or files created.

            - metrics_module. String identifying the module containing
            the metrics.

            - params. Optional. Dictionary containing the parameters
            that the user whishes to specify.

            - verbose. Boolean specifying whether output must be
            produced in-line.

        """
        self.verbose = verbose
        
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        
        """设置参数"""
        # 全局参数
        self.PREFIX = name
        # if 'WGAN' in params:
        #     self.WGAN = params['WGAN']
        # else:
        #     self.WGAN = False

        if 'SEED' in params:
            self.SEED = params['SEED']
        else:
            self.SEED = None
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        if 'BATCH_SIZE' in params:
            self.BATCH_SIZE = params['BATCH_SIZE']
        else:
            self.BATCH_SIZE = 64
        if 'TOTAL_BATCH' in params:
            self.TOTAL_BATCH = params['TOTAL_BATCH']
        else:
            self.TOTAL_BATCH = 200
        if 'GENERATED_NUM' in params:
            self.GENERATED_NUM = params['GENERATED_NUM']
        else:
            self.GENERATED_NUM = 10000

        if 'VOCAB_SIZE' in params:
            self.VOCAB_SIZE = params['VOCAB_SIZE']
        else:        
            self.VOCAB_SIZE = 5000
        if 'PRE_EPOCH_NUM' in params:
            self.PRE_EPOCH_NUM = 120
        else:
            self.NUM_CLASS = 2

        if 'POSITIVE_FILE' in params:
            self.POSITIVE_FILE = params['POSITIVE_FILE']
        else:
            raise ValueError('Positive file not specified.')
        if 'NEGATIVE_FILE' in params:
            self.NEGATIVE_FILE = 'gene.data'
        else:
            raise ValueError('Negative file not specified.')
        
        #Generator参数
        if 'g_emb_dim' in params:
            self.g_emb_dim = params['g_emb_dim']
        else:
            self.g_emb_dim = 32
        if 'g_hidden_dim' in params:
            self.g_hidden_dim = params['g_hidden_dim']
        else:
            self.g_hidden_dim = 32
        if 'g_sequence_len' in params:
            self.g_sequence_len = params['g_sequence_len']
        else:
            self.g_sequence_len = 20

        #Discriminator参数
        if 'd_emb_dim' in params:
            self.d_emb_dim = params['d_emb_dim']
        else:
            self.d_emb_dim = 64
        if 'd_num_classes' in params:
            self.d_num_classes = params['d_num_classes']
        else:
            raise ValueError('Number of classes not specified.')
        if 'd_filter_sizes' in params:
            self.d_filter_sizes = params['d_filter_sizes']
        else:
            self.d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        if 'd_num_filters' in params:
            self.d_num_filters = params['d_num_filters']
        else:
            self.d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        if 'd_dropout' in params:
            self.d_dropout = params['d_dropout']
        else:
            self.d_dropout = 0.75
        if 'd_l2reg' in params:
            self.d_l2reg = params['d_l2reg']
        else:
            self.d_l2reg = 0.2
        
        global mm
        if metrics_module == 'mol_metrics':
            mm = mol_metrics
        else:
            raise ValueError('Metrics module not recognized.')
        
        self.AV_METRICS = mm.get_metrics()
        self.LOADINGS = mm.metrics_loading()

        # self.PRETRAINED = False

    def load_training_set(self, file):
        """Specifies a training set for the model. It also finishes
        the model set up, as some of the internal parameters require
        knowledge of the vocabulary.

        Arguments
        -----------

            - file. String pointing to the dataset file.

        """

        # Load training set
        self.train_samples = mm.load_train_data(file)

        # Process and create vocabulary
        self.char_dict, self.ord_dict = mm.build_vocab(self.train_samples)
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = self.ord_dict[self.NUM_EMB - 1]
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.train_samples))
        print('Vocabulary:')
        print(list(self.char_dict.keys()))
        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.train_samples, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if mm.verified_and_below(sample, self.MAX_LENGTH)]
        self.positive_samples = [mm.encode(sam,
                                           self.MAX_LENGTH,
                                           self.char_dict) for sam in to_use]
        self.POSITIVE_NUM = len(self.positive_samples)

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            lens = [len(s) for s in to_use]
            print('Avg Length to use is     :   {:2.2f} ({:2.2f}) [{:d},{:d}]'.format(
                np.mean(lens), np.std(lens), np.min(lens), np.max(lens)))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Size of alphabet is      :   {}'.format(self.NUM_EMB))
            print('')

            # params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
            #           'GEN_ITERATIONS', 'GEN_BATCH_SIZE', 'SEED',
            #           'DIS_BATCH_SIZE', 'DIS_EPOCHS', 'EPOCH_SAVES',
            #           'CHK_PATH', 'GEN_EMB_DIM', 'GEN_HIDDEN_DIM',
            #           'START_TOKEN', 'SAMPLE_NUM', 'BIG_SAMPLE_NUM',
            #           'LAMBDA', 'MAX_LENGTH', 'DIS_EMB_DIM',
            #           'DIS_FILTER_SIZES', 'DIS_NUM_FILTERS',
            #           'DIS_DROPOUT', 'DIS_L2REG']
            
            params = ['SEED', 'BATCH_SIZE', 'TOTAL_BATCH', 
                      'GENERATED_NUM', 'VOCAB_SIZE', 'PRE_EPOCH_NUM', 
                      'd_emb_num', 'd_num_classes', 'd_filter_sizes',
                      'd_num_filters', 'd_dropout', 'd_l2reg', 
                      'g_emb_dim','g_hidden_dim', 'g_sequence_len', 
                      'POSITIVE_FILE', 'NEGATIVE_FILE',
                      'CHK_PATH', 'START_TOKEN', 'MAX_LENGTH'
                      'SAMPLE_NUM', 'BIG_SAMPLE_NUM', 'LAMBDA']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = GenDataIter(self.GEN_BATCH_SIZE)
        self.dis_loader = DisDataIter()
        self.mle_loader = GenDataIter(self.GEN_BATCH_SIZE)
        # if self.WGAN:
        #     self.generator = WGenerator(self.NUM_EMB, self.GEN_BATCH_SIZE,
        #                                 self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
        #                                 self.MAX_LENGTH, self.START_TOKEN)
        #     self.discriminator = WDiscriminator(
        #         sequence_length=self.MAX_LENGTH,
        #         num_classes=2,
        #         vocab_size=self.NUM_EMB,
        #         embedding_size=self.DIS_EMB_DIM,
        #         filter_sizes=self.DIS_FILTER_SIZES,
        #         num_filters=self.DIS_NUM_FILTERS,
        #         l2_reg_lambda=self.DIS_L2REG,
        #         wgan_reg_lambda=self.WGAN_REG_LAMBDA,
        #         grad_clip=self.DIS_GRAD_CLIP)
        # else:
        self.generator = Generator(self.NUM_EMB, self.g_emb_dim,
                                   self.g_hidden_dim, self.NUM_CLASS, self.cuda,
                                   self.MAX_LENGTH, self.START_TOKEN)
        self.discriminator = Discriminator(
            sequence_length=self.MAX_LENGTH,
            num_classes=2,
            vocab_size=self.NUM_EMB,
            embedding_size=self.DIS_EMB_DIM,
            filter_sizes=self.DIS_FILTER_SIZES,
            num_filters=self.DIS_NUM_FILTERS,
            l2_reg_lambda=self.DIS_L2REG,
            grad_clip=self.DIS_GRAD_CLIP)
        
        # # Set up PyTorch training
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.generator.to(self.device)
        # self.discriminator.to(self.device)

        # # Initialize optimizers
        # self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        # self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

        # # Initialize loss function
        # self.criterion = nn.BCELoss()

        # TensorBoard setup (if needed)
        # if 'TBOARD_LOG' in params:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.tb_writer = SummaryWriter(log_dir=self.CHK_PATH)

    def train_epoch(self, model, data_iter, criterion, optimizer):
        total_loss = 0.
        total_words = 0.
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_iter.reset()
        return math.exp(total_loss / total_words)

    def eval_epoch(self, model, data_iter, criterion):
        total_loss = 0.
        total_words = 0.
        with torch.no_grad():
            for (data, target) in data_iter:#tqdm(
                #data_iter, mininterval=2, desc=' - Training', leave=False):
                data = Variable(data)
                target = Variable(target)
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                target = target.contiguous().view(-1)
                pred = model.forward(data)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_words += data.size(0) * data.size(1)
            data_iter.reset()

        assert total_words > 0  # Otherwise NullpointerException
        return math.exp(total_loss / total_words)
    
    