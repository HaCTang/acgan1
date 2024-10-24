# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch
class GenDataIter(object):
    """Data iterator to load sample."""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_batches(self, samples):
        self.num_batch = int(len(samples) / self.batch_size)
        samples = samples[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(samples), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
    # def __init__(self, data_files, batch_size):
    #     super(GenDataIter, self).__init__()
    #     self.batch_size = batch_size
    #     self.data_lis, self.labels = self.read_files(data_files)
    #     self.data_num = len(self.data_lis)
    #     # self.indices = range(self.data_num) # Python 2, hard to shuffle
    #     self.indices = list(range(self.data_num))
    #     self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
    #     self.idx = 0

    # def __len__(self):
    #     return self.num_batches

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     return self.next()

    # def reset(self):
    #     self.idx = 0
    #     combined = list(zip(self.data_lis, self.labels))
    #     random.shuffle(combined)
    #     self.data_lis, self.labels = zip(*combined)

    # def next(self):
    #     if self.idx >= self.data_num:
    #         raise StopIteration
    #     index = self.indices[self.idx:self.idx+self.batch_size]
    #     d = [self.data_lis[i] for i in index]
    #     labels = [self.labels[i] for i in index]
    #     d = torch.LongTensor(np.asarray(d, dtype='int64'))
    #     labels = torch.LongTensor(np.asarray(labels, dtype='int64'))
    #     data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
    #     target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
    #     self.idx += self.batch_size
    #     return data, target, labels

    # def read_files(self, data_files: list):
    #     """
    #     Read multiple files for data, each file represents a different class.

    #     @param data_files: List of file paths, e.g., ["class0_data.txt", "class1_data.txt", ...]
    #     @return: Tuple containing a list of data and a list of corresponding labels.
    #     """
    #     data_list = []
    #     labels = []
    #     for label, file in enumerate(data_files):
    #         """
    #         label 会被自动设置为 0, 1, 2, ...，
    #         分别对应 class0_data.txt、class1_data.txt 等文件的类别。
    #         """
    #         with open(file, 'r') as f:
    #             lines = f.readlines()
    #         for line in lines:
    #             l = line.strip().split(' ')
    #             l = [int(s) for s in l]
    #             data_list.append(l)
    #             labels.append(label)
    #     return data_list, labels

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self):
        self.vocab_size = 5000

    def load_data_and_labels(self, positive_examples, negative_examples):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Split by words
        x_text = positive_examples + negative_examples

        # Generate labels
        positive_labels = [[0,] for _ in positive_examples]
        negative_labels = [[1,] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        x_text = np.array(x_text)
        y = np.array(y)
        return [x_text, y]

    def load_train_data(self, positive_file, negative_file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        classified_sentences, labels = self.load_data_and_labels(
            positive_file, negative_file)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = classified_sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        self.sequence_length = 20 
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])

        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([0, 1])

        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(list(data))
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    # def __init__(self, real_data_files, fake_data_file, batch_size):
    #     super(DisDataIter, self).__init__()
    #     self.batch_size = batch_size
    #     real_data_lis, real_labels = self.read_files(real_data_files)
    #     fake_data_lis = self.read_file(fake_data_file)
    #     fake_labels = [len(real_data_files)] * len(fake_data_lis)
    #     self.data = real_data_lis + fake_data_lis
    #     self.labels = real_labels + fake_labels
    #     self.pairs = list(zip(self.data, self.labels))
    #     self.data_num = len(self.pairs)
    #     self.indices = list(range(self.data_num))
    #     self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
    #     self.idx = 0

    # def __len__(self):
    #     return self.num_batches

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     return self.next()

    # def reset(self):
    #     self.idx = 0
    #     random.shuffle(self.pairs)

    # def next(self):
    #     if self.idx >= self.data_num:
    #         raise StopIteration
    #     index = self.indices[self.idx:self.idx+self.batch_size]
    #     pairs = [self.pairs[i] for i in index]
    #     data = [p[0] for p in pairs]
    #     labels = [p[1] for p in pairs]
    #     data = torch.LongTensor(np.asarray(data, dtype='int64'))
    #     labels = torch.LongTensor(np.asarray(labels, dtype='int64'))
    #     self.idx += self.batch_size
    #     return data, labels

    # def read_files(self, data_files):
    #     """
    #     Read multiple files for data, each file represents a different class.

    #     @param data_files: List of file paths, e.g., ["class0_data.txt", "class1_data.txt", ...]
    #     @return: Tuple containing a list of data and a list of corresponding labels.
    #     """
    #     data_list = []
    #     labels = []
    #     for label, file in enumerate(data_files):
    #         """
    #         label 会被自动设置为 0, 1, 2, ...，
    #         分别对应 class0_data.txt、class1_data.txt 等文件的类别。
    #         """
    #         with open(file, 'r') as f:
    #             lines = f.readlines()
    #         for line in lines:
    #             l = line.strip().split(' ')
    #             l = [int(s) for s in l]
    #             data_list.append(l)
    #             labels.append(label)
    #     return data_list, labels

    # def read_file(self, data_file):
    #     """
    #     Read a single file for fake data.
    #     """
    #     with open(data_file, 'r') as f:
    #         lines = f.readlines()
    #     data_list = []
    #     for line in lines:
    #         l = line.strip().split(' ')
    #         l = [int(s) for s in l]
    #         data_list.append(l)
    #     return data_list, None
