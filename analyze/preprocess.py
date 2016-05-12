__author__ = 'shengjia'


import numpy as np
import os
from config import settings


class Converter:
    def __init__(self, layer_list):
        self.root_dir = settings['activation_root']
        self.layer_list = layer_list

    def output_binary(self, top_K, replace=False):
        top_K = 10

        for layer in self.layer_list:
            if not replace and os.path.isfile(os.path.join(self.root_dir, 'binary_' + str(top_K) + '_' + layer + '.npy')):
                continue
            print("Transforming layer " + layer)
            activation = np.load(os.path.join(self.root_dir, layer + '.npy'))

            # Code for testing correctness
            # activation = np.array([[9, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1], [2, 5, 7, 8, 2, 1, 8, 2, 8]], np.float)
            # top_K = 2
            # Normalize activation
            min_mat = np.tile(np.expand_dims(np.min(activation, 0), 0), (activation.shape[0], 1))
            range_mat = np.max(activation, 0) - np.min(activation, 0)
            range_mat = np.clip(range_mat, 1e-6, np.max(range_mat))
            range_mat = np.tile(np.expand_dims(range_mat, 0), (activation.shape[0], 1))
            activation = np.divide(np.subtract(activation, min_mat), range_mat)
            # print(activation)
            for index in range(activation.shape[0]):
                max_index = activation[index, :].argsort()[-top_K:]
                activation[index, :] = np.zeros(activation.shape[1], np.float)
                for max in max_index:
                    activation[index, max] = 1.0
                if index % 1000 == 0:
                    print("Processing " + str(index) + "-th image")
            np.save(os.path.join(self.root_dir, 'binary_' + str(top_K) + '_' + layer), activation)

    def output_covariance(self, top_K):
        self.output_binary(top_K)
        activations = []
        print("Loading activations")
        for layer in self.layer_list:
            activations.append(np.load(os.path.join(self.root_dir, 'binary_' + str(top_K) + '_' + layer + '.npy')))
        print("Computing convariance matrix")
        total = np.concatenate(activations, 1)
        cov = np.dot(total.transpose(), total)
        print("Writing to file")
        np.save(os.path.join(self.root_dir, 'covariance_' + str(top_K)), cov)
        mat_sum = np.sum(cov)
        print("Sum of elements is " + str(mat_sum) + " in a matrix with " + str(cov.size) + " elements")


if __name__ == '__main__':
    converter = Converter(layer_list = ['conv3', 'conv4', 'conv5'])
    converter.output_binary(10, replace=True)
    converter.output_covariance(10)