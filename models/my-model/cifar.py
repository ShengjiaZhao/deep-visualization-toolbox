__author__ = 'shengjia'


import cPickle
import os
import numpy as np


class DatasetCIFAR:
    def __init__(self, cifar_path='../../cifar/'):
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
            self.data_array[batch_size*i:batch_size*(i+1), :] = batches[i]['data'].astype(np.float32) / 256
            labels = batches[i]['labels']
            for j in range(batch_size):
                self.label_array[batch_size*i:batch_size*(i+1)+j, labels[i]] = 1.0

        test_batch_size = test_batch['data'].shape[0]
        self.test_data_array = test_batch['data'].astype(np.float32) / 256.0
        self.test_label_array = np.zeros((test_batch_size, 10), np.float32)
        for j in range(test_batch_size):
            self.test_label_array[test_batch['labels'][j]] = 1.0

    @staticmethod
    def convert_to_rgb(arr):
        res = np.ndarray((32, 32, 3), np.float32)
        res[:, :, 0] = np.reshape(arr[:1024], (32, 32))
        res[:, :, 1] = np.reshape(arr[1024:2048], (32, 32))
        res[:, :, 2] = np.reshape(arr[2048:], (32, 32))
        return res

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