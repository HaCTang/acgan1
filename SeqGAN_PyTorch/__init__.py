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
from collections import Counter

from SeqGAN_PyTorch.data_iter import GenDataIter, DisDataIter
from SeqGAN_PyTorch.generator import Generator
from SeqGAN_PyTorch.discriminator import Discriminator
from SeqGAN_PyTorch.rollout import Rollout

# from rdkit import rdBASE
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

        if 'PRETRAIN_GEN_EPOCHS' in params:
            self.PRETRAIN_GEN_EPOCHS = params['PRETRAIN_GEN_EPOCHS']
        else:
            self.PRETRAIN_GEN_EPOCHS = 240

        if 'PRETRAIN_DIS_EPOCHS' in params:
            self.PRETRAIN_DIS_EPOCHS = params['PRETRAIN_DIS_EPOCHS']
        else:
            self.PRETRAIN_DIS_EPOCHS = 50

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
        if 'GEN_BATCH_SIZE' in params:
            self.GEN_BATCH_SIZE = params['GEN_BATCH_SIZE']
        else:
            self.GEN_BATCH_SIZE = 64
        if 'DIS_BATCH_SIZE' in params:
            self.DIS_BATCH_SIZE = params['DIS_BATCH_SIZE']
        else:
            self.DIS_BATCH_SIZE = 64
        if 'GENERATED_NUM' in params:
            self.GENERATED_NUM = params['GENERATED_NUM']
        else:
            self.GENERATED_NUM = 10000

        if 'VOCAB_SIZE' in params:
            self.VOCAB_SIZE = params['VOCAB_SIZE']
        else:        
            self.VOCAB_SIZE = 5000
        if 'PRE_EPOCH_NUM' in params:
            self.PRE_EPOCH_NUM = params['PRE_EPOCH_NUM']
        else:
            self.PRE_EPOCH_NUM = 120
        if 'NUM_CLASS' in params:
            self.NUM_CLASS = params['NUM_CLASS']
        else:
            self.NUM_CLASS = 2
        
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
        if 'g_iterations' in params:
            self.g_iterations = params['g_iterations']
        else:
            self.g_iterations = 2

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
        if 'd_grad_clip' in params:
            self.d_grad_clip = params['d_grad_clip']
        else:
            self.d_grad_clip = 1.0
        if 'd_l2reg' in params:
            self.d_l2reg = params['d_l2reg']
        else:
            self.d_l2reg = 0.2
        
        #先验混合模型参数
        if 'LAMBDA_1' in params:
            self.LAMBDA_1 = params['LAMBDA_1']
        else:
            self.LAMBDA_1 = 0.5
        if 'LAMBDA_2' in params:
            self.LAMBDA_2 = params['LAMBDA_2']
        else:
            self.LAMBDA_2 = 0.5

        #其它参数
        if 'START_TOKEN' in params:
            self.START_TOKEN = params['START_TOKEN']
        else:
            self.START_TOKEN = 0
        if 'MAX_LENGTH' in params:
            self.MAX_LENGTH = params['MAX_LENGTH']
        if 'CHK_PATH' in params:
            self.CHK_PATH = params['CHK_PATH']
        else:
            self.CHK_PATH = os.path.join(
                os.getcwd(), 'checkpoints\{}'.format(self.PREFIX))

        global mm
        if metrics_module == 'mol_metrics':
            mm = mol_metrics
        else:
            raise ValueError('Metrics module not recognized.')
        
        self.AV_METRICS = mm.get_metrics()
        self.LOADINGS = mm.metrics_loading()

        self.PRETRAINED = False

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
        self.molecules, _ = zip(*self.train_samples) # Extract molecules
        self.DATA_LENGTH = max(map(len, self.molecules))
        print('Vocabulary:')
        print(list(self.char_dict.keys()))
        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.molecules, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if mm.verified_and_below(sample[0], self.MAX_LENGTH)]
        # self.positive_samples1 = [mm.encode(sam[0],
        #                                    self.MAX_LENGTH,
        #                                    self.char_dict) for sam in to_use]
        molecules_to_use, label_to_use = zip(*to_use)
        positive_molecules = [mm.encode(sam,
                            self.MAX_LENGTH,
                            self.char_dict) for sam in molecules_to_use]
        self.positive_samples = [list(item) for item in zip(positive_molecules, label_to_use)]
        # print("positive_samples:", self.positive_samples)

        self.POSITIVE_NUM = len(self.positive_samples)
        self.TYPE_NUM = Counter([sam[1] for sam in to_use]) # Number of samples per type

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            lens = [len(s[0]) for s in to_use]
            print('Avg Length to use is     :   {:2.2f} ({:2.2f}) [{:d},{:d}]'.format(
                np.mean(lens), np.std(lens), np.min(lens), np.max(lens)))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Num different samples is :   {}'.format(
                self.TYPE_NUM))
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
            
            params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
                      'SEED', 'BATCH_SIZE', 'TOTAL_BATCH', 
                      'GEN_BATCH_SIZE', 'DIS_BATCH_SIZE', 'NUM_CLASS',
                      'GENERATED_NUM', 'VOCAB_SIZE', 'PRE_EPOCH_NUM', 
                      'd_num_classes', 'd_filter_sizes',
                      'd_num_filters', 'd_dropout', 'd_l2reg', 'd_grad_clip',
                      'g_emb_dim','g_hidden_dim', 'g_sequence_len', 'g_iterations',
                      'CHK_PATH', 'START_TOKEN', 'MAX_LENGTH', 
                      'LAMBDA_1', 'LAMBDA_2']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = GenDataIter(self.BATCH_SIZE)
        self.dis_loader = DisDataIter()
        self.mle_loader = GenDataIter(self.BATCH_SIZE)  # For MLE training, 暂时没用
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
        self.generator = Generator(self.NUM_EMB, self.BATCH_SIZE, self.g_emb_dim,
                                   self.g_hidden_dim, self.NUM_CLASS, self.cuda,
                                   self.MAX_LENGTH, self.START_TOKEN)
        self.discriminator = Discriminator(
            sequence_length=self.MAX_LENGTH,
            num_classes=2,
            vocab_size=self.NUM_EMB,
            emb_dim=self.d_emb_dim,
            filter_sizes=self.d_filter_sizes,
            num_filters=self.d_num_filters,
            l2_reg_lambda=self.d_l2reg,
            grad_clip=self.d_grad_clip)
        
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

# todo: 以下函数未完成，需要再加一个参数，metrics用于classlabel的评价指标
    def set_training_program(self, metrics=None, steps=None):
        """Sets a program of metrics and epochs
        for training the model and generating molecules.

        Arguments
        -----------

            - metrics. List of metrics. Each element represents
            the metric used with a particular set of epochs. Its
            length must coincide with the steps list.

            - steps. List of epoch sets. Each element represents
            the number of epochs for which a given metric will
            be used. Its length must coincide with the steps list.

        Note
        -----------

            The program will crash if both lists have different
            lengths.

        """

        # Raise error if the lengths do not match
        if len(metrics) != len(steps):
            return ValueError('Unmatching lengths in training program.')

        # Set important parameters
        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.METRICS = metrics

        # Build the 'educative program'
        self.EDUCATION = {}
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

# todo: 以下函数未完成，需要再加一个参数，metrics用于classlabel的评价指标
    def load_metrics(self):
        """Loads the metrics."""

        # Get the list of used metrics
        met = list(set(self.METRICS))

        # Execute the metrics loading
        self.kwargs = {}
        for m in met:
            load_fun = self.LOADINGS[m]
            args = load_fun()
            if args is not None:
                if isinstance(args, tuple):
                    self.kwargs[m] = {args[0]: args[1]}
                elif isinstance(args, list):
                    fun_args = {}
                    for arg in args:
                        fun_args[arg[0]] = arg[1]
                    self.kwargs[m] = fun_args
            else:
                self.kwargs[m] = None

    def load_prev_pretraining(self, ckpt=None):
        """
        Loads a previous pretraining.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name_pretrain/pretrain_ckpt' is assumed.

        Note
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files, like in the following ls:

                checkpoint
                pretrain_ckpt.data-00000-of-00001
                pretrain_ckpt.index
                pretrain_ckpt.meta

            In this case, ckpt = 'pretrain_ckpt'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        # Load from checkpoint
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.generator.optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            self.discriminator.optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt))
            self.PRETRAINED = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt))

    def load_prev_training(self, ckpt=None):
        """
        Loads a previous trained model.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name/pretrain_ckpt' is assumed.

        Note 1
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files. An example ls:

                checkpoint
                validity_model_0.ckpt.data-00000-of-00001
                validity_model_0.ckpt.index
                validity_model_0.ckpt.meta
                validity_model_100.ckpt.data-00000-of-00001
                validity_model_100.ckpt.index
                validity_model_100.ckpt.meta
                validity_model_120.ckpt.data-00000-of-00001
                validity_model_120.ckpt.index
                validity_model_120.ckpt.meta
                validity_model_140.ckpt.data-00000-of-00001
                validity_model_140.ckpt.index
                validity_model_140.ckpt.meta

                    ...

                validity_model_final.ckpt.data-00000-of-00001
                validity_model_final.ckpt.index
                validity_model_final.ckpt.meta

            Possible ckpt values are 'validity_model_0', 'validity_model_140'
            or 'validity_model_final'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # If there is no Rollout, add it
        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.generator.optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            self.discriminator.optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            print('Training loaded from previous checkpoint {}'.format(ckpt))
            self.SESS_LOADED = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt))

    def pretrain(self):
        """Pretrains generator and discriminator."""
        # print(self.positive_samples)
        self.gen_loader.create_batches(self.positive_samples)
        # results = OrderedDict({'exp_name': self.PREFIX})

        if self.verbose:
            print('\nPRETRAINING')
            print('============================\n')
            print('GENERATOR PRETRAINING')

        t_bar = trange(self.PRETRAIN_GEN_EPOCHS) #创建一个进度条
        for epoch in t_bar:
            supervised_g_losses = []
            self.gen_loader.reset_pointer()
            for it in range(self.gen_loader.num_batch):
                batch = self.gen_loader.next_batch()
                x, class_label = self.gen_loader.batch_to_tensor(batch)
                g_loss = self.generator.pretrain_step(x, class_label)
                supervised_g_losses.append(g_loss)
            # print results
            mean_g_loss = np.mean(supervised_g_losses)
            t_bar.set_postfix(G_loss=mean_g_loss)

        samples = self.generate_samples(self.GENERATED_NUM)
        self.mle_loader.create_batches(samples) # For MLE training, 暂时没用

        if self.LAMBDA_1 != 0:

            if self.verbose:
                print('\nDISCRIMINATOR PRETRAINING')
            t_bar = trange(self.PRETRAIN_DIS_EPOCHS)
            for i in t_bar:
                negative_samples = self.generate_samples(self.POSITIVE_NUM)
                dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                    self.positive_samples, negative_samples)
                dis_batches = self.dis_loader.batch_iter(
                    zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE,
                    self.PRETRAIN_DIS_EPOCHS)
                supervised_d_losses = []

                for batch in dis_batches:                    
                    x_batch, y_batch = zip(*batch)
                    x, x_label = zip(*x_batch)

                    # Convert to tensors
                    x = torch.tensor(x, dtype=torch.long)
                    y_batch = torch.tensor(np.array(y_batch), dtype=torch.float)
                    x_label = torch.tensor(x_label, dtype=torch.int64)

                    d_loss = self.discriminator.train_step(x, y_batch, x_label)
                    supervised_d_losses.append(d_loss)

                # print results
                mean_d_loss = np.mean(supervised_d_losses)
                t_bar.set_postfix(D_loss=mean_d_loss)

        self.PRETRAINED = True
        return
    
    def generate_samples(self, num):
        """Generates molecules. Returns a list of samples, the same shape of self.positive_samples.

        Arguments
        -----------

            - num. Integer representing the number of molecules

        """

        generated_samples = []

        for _ in range(int(num / self.GEN_BATCH_SIZE)):
            for class_label in range(self.NUM_CLASS):
                class_label_tensor = torch.tensor([class_label] * self.GEN_BATCH_SIZE, dtype=torch.int64)
                gen_x, _ = self.generator.generate(class_label_tensor)
                for i in range(self.GEN_BATCH_SIZE):
                    generated_samples.append([gen_x[i].tolist(), class_label])

        return generated_samples

    def report_rewards(self, rewards, metric):
        print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
        print('Reward: {}  (lambda={:.2f})'.format(metric, self.LAMBDA))
        #np.set_printoptions(precision=3, suppress=True)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        min_r, max_r = np.min(rewards), np.max(rewards)
        print('Stats: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
            mean_r, std_r, min_r, max_r))
        non_neg = rewards[rewards > 0.01]
        if len(non_neg) > 0:
            mean_r, std_r = np.mean(non_neg), np.std(non_neg)
            min_r, max_r = np.min(non_neg), np.max(non_neg)
            print('Valid: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
                mean_r, std_r, min_r, max_r))
        #np.set_printoptions(precision=8, suppress=False)
        return

    def train(self, ckpt_dir='checkpoints/'):
        """Trains the model. If necessary, also includes pretraining."""

        if not self.PRETRAINED:
            self.pretrain()

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir, '{}_pretrain_ckpt.pth'.format(self.PREFIX))
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'gen_optimizer_state_dict': self.generator.optimizer.state_dict(),
                'dis_optimizer_state_dict': self.discriminator.optimizer.state_dict()
            }, ckpt_file)
            if self.verbose:
                print('Pretrain saved at {}'.format(ckpt_file))

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)
            self.rollout.g_embeddings = self.generator.emb

        if self.verbose:
            print('\nSTARTING TRAINING')
            print('============================\n')

        results_rows = []
        losses = defaultdict(list)
        for nbatch in tqdm(range(self.TOTAL_BATCH)):

            results = OrderedDict({'exp_name': self.PREFIX})
            metric = self.EDUCATION[nbatch]

            if metric in self.AV_METRICS.keys():
                reward_func = self.AV_METRICS[metric]
            else:
                raise ValueError('Metric {} not found!'.format(metric))

            if self.kwargs[metric] is not None:

                def batch_reward(samples, train_samples=None):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples,
                                          **self.kwargs[metric])
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            else:

                def batch_reward(samples, train_samples=None):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples)
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            if nbatch % 10 == 0:
                gen_samples = self.generate_samples(5*self.GENERATED_NUM)
            else:
                gen_samples = self.generate_samples(self.GENERATED_NUM)
            self.gen_loader.create_batches(gen_samples)
            results['Batch'] = nbatch
            print('Batch n. {}'.format(nbatch))
            print('============================\n')
            print('\nGENERATOR TRAINING')
            print('============================\n')        

            mm.compute_results(batch_reward,
                               gen_samples, self.train_samples, self.ord_dict, results)

            for it in range(self.g_iterations):
                for i in range(self.NUM_CLASS):
                    samples, sample_labels = self.generator.generate(torch.tensor([i] * self.GEN_BATCH_SIZE))
                    # rewards = self.rollout.get_reward(samples, sample_labels, 16, self.discriminator, 
                    #                                 batch_reward, self.LAMBDA_1)
                    rewards = self.rollout.get_reward(samples, sample_labels, 1, self.discriminator, 
                                                    batch_reward, self.LAMBDA_1)

                    g_loss = self.generator.train_step(samples, sample_labels, rewards)
                    losses['G-loss'].append(g_loss)
                    self.generator.g_count += 1
                    self.report_rewards(rewards, metric)

            self.rollout.update_params()

            # generate for discriminator
            if self.LAMBDA_1 != 0:
                print('\nDISCRIMINATOR TRAINING')
                print('============================\n')
                for i in range(self.DIS_EPOCHS):
                    print('Discriminator epoch {}...'.format(i + 1))

                    negative_samples = self.generate_samples(self.POSITIVE_NUM)
                    dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                        self.positive_samples, negative_samples)
                    dis_batches = self.dis_loader.batch_iter(
                        zip(dis_x_train, dis_y_train),
                        self.DIS_BATCH_SIZE, self.DIS_EPOCHS
                    )

                    d_losses, ce_losses, l2_losses, w_loss = [], [], [], []
                    for batch in dis_batches:
                        x_batch, y_batch = zip(*batch)
                        d_loss, ce_loss, l2_loss, w_loss = self.discriminator.train_step(
                            x_batch, y_batch, self.DIS_DROPOUT)
                        d_losses.append(d_loss)
                        ce_losses.append(ce_loss)
                        l2_losses.append(l2_loss)

                    losses['D-loss'].append(np.mean(d_losses))
                    losses['CE-loss'].append(np.mean(ce_losses))
                    losses['L2-loss'].append(np.mean(l2_losses))
                    losses['WGAN-loss'].append(np.mean(l2_losses))

                    self.discriminator.d_count += 1

                print('\nDiscriminator trained.')

            results_rows.append(results)

            # save model
            if nbatch % self.EPOCH_SAVES == 0 or \
               nbatch == self.TOTAL_BATCH - 1:

                if results_rows is not None:
                    df = pd.DataFrame(results_rows)
                    df.to_csv('{}_results.csv'.format(
                        self.PREFIX), index=False)
                for key, val in losses.items():
                    v_arr = np.array(val)
                    np.save('{}_{}.npy'.format(self.PREFIX, key), v_arr)

                if nbatch is None:
                    label = 'final'
                else:
                    label = str(nbatch)

                # save models
                ckpt_dir = self.CHK_PATH

                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_file = os.path.join(ckpt_dir, '{}_{}.pth'.format(self.PREFIX, label))
                torch.save({
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'gen_optimizer_state_dict': self.generator.optimizer.state_dict(),
                    'dis_optimizer_state_dict': self.discriminator.optimizer.state_dict()
                }, ckpt_file)
                print('\nModel saved at {}'.format(ckpt_file))

        print('\n######### FINISHED #########')



    # def train_epoch(self, model, data_iter, criterion, optimizer):
    #     total_loss = 0.
    #     total_words = 0.
    #     for (data, target) in data_iter:#tqdm(
    #         #data_iter, mininterval=2, desc=' - Training', leave=False):
    #         data = Variable(data)
    #         target = Variable(target)
    #         if self.cuda:
    #             data, target = data.cuda(), target.cuda()
    #         target = target.contiguous().view(-1)
    #         pred = model.forward(data)
    #         loss = criterion(pred, target)
    #         total_loss += loss.item()
    #         total_words += data.size(0) * data.size(1)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     data_iter.reset()
    #     return math.exp(total_loss / total_words)

    # def eval_epoch(self, model, data_iter, criterion):
    #     total_loss = 0.
    #     total_words = 0.
    #     with torch.no_grad():
    #         for (data, target) in data_iter:#tqdm(
    #             #data_iter, mininterval=2, desc=' - Training', leave=False):
    #             data = Variable(data)
    #             target = Variable(target)
    #             if self.cuda:
    #                 data, target = data.cuda(), target.cuda()
    #             target = target.contiguous().view(-1)
    #             pred = model.forward(data)
    #             loss = criterion(pred, target)
    #             total_loss += loss.item()
    #             total_words += data.size(0) * data.size(1)
    #         data_iter.reset()

    #     assert total_words > 0  # Otherwise NullpointerException
    #     return math.exp(total_loss / total_words)
    
    