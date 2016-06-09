__author__ = 'shengjia'


import cPickle
import os
import numpy as np


class DatasetCIFAR:
    def __init__(self, cifar_path='../../cifar/', gray_scale=False):
        self.gray_scale = gray_scale
        batches = []
        for batch_index in range(1, 6):
            file_path = os.path.join(cifar_path, 'data_batch_' + str(batch_index))
            batches.append(self.unpickle(file_path))
        self.label_names = self.unpickle(os.path.join(cifar_path, 'batches.meta'))['label_names']
        test_batch = self.unpickle(os.path.join(cifar_path, 'test_batch'))
        batch_size = 10000

        self.data_array = np.ndarray((batch_size*5, 1024*3), np.float32)
        self.label_array = np.zeros((batch_size*5, 10), np.float32)
        for i in range(5):
            self.data_array[batch_size*i:batch_size*(i+1), :] = batches[i]['data'].astype(np.float32) / 256.0
            labels = batches[i]['labels']
            for j in range(batch_size):
                self.label_array[batch_size*i+j, labels[j]] = 1.0

        test_batch_size = test_batch['data'].shape[0]
        self.test_data_array = test_batch['data'].astype(np.float32) / 256.0
        self.test_label_array = np.zeros((test_batch_size, 10), np.float32)
        for j in range(test_batch_size):
            self.test_label_array[j, test_batch['labels'][j]] = 1.0

        self.train_count = batch_size * 5
        self.test_count = test_batch_size
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0

        if self.gray_scale:
            self.data_array = (self.data_array[:, :1024] + self.data_array[:, 1024:2048] + self.data_array[:, 2048:]) / 3.0
            self.test_data_array = (self.test_data_array[:, :1024] + self.test_data_array[:, 1024:2048] + self.test_data_array[:, 2048:]) / 3.0

    @staticmethod
    def convert_to_rgb(arr):
        res = np.ndarray((32, 32, 3), np.float32)
        res[:, :, 0] = np.reshape(arr[:1024], (32, 32))
        res[:, :, 1] = np.reshape(arr[1024:2048], (32, 32))
        res[:, :, 2] = np.reshape(arr[2048:], (32, 32))
        return res

    @staticmethod
    def convert_to_gray(arr):
        return np.reshape(arr, (32, 32))

    @staticmethod
    def unpickle(file):
        if os.path.isfile(file):
            fo = open(file, 'rb')
            dict = cPickle.load(fo)
            fo.close()
            return dict
        else:
            print("Cifar file " + str(file) + " not found")
            exit(-1)

    def next_batch(self, batch_size):
        if self.train_batch_ptr + batch_size > self.train_count:
            self.train_batch_ptr = 0

        self.train_batch_ptr += batch_size
        return (self.data_array[self.train_batch_ptr-batch_size:self.train_batch_ptr],
                self.label_array[self.train_batch_ptr-batch_size:self.train_batch_ptr])

    def next_test_batch(self, batch_size):
        if self.test_batch_ptr + batch_size > self.test_count:
            self.test_batch_ptr = 0

        self.test_batch_ptr += batch_size
        return (self.test_data_array[self.test_batch_ptr-batch_size:self.test_batch_ptr],
                self.test_label_array[self.test_batch_ptr-batch_size:self.test_batch_ptr])
