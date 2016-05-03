__author__ = 'shengjia'


import numpy as np
import os

class Converter:
    def __init__(self, root_dir, layer_list):
        self.root_dir = root_dir
        self.layer_list = layer_list

    def output_binary(self, top_K, replace=False):
        top_K = 10

        for layer in self.layer_list:
            if not replace and os.path.isfile(os.path.join(self.root_dir, 'binary_' + str(top_K) + '_' + layer + '.npy')):
                continue
            print("Transforming layer " + layer)
            activation = np.load(os.path.join(self.root_dir, layer + '.npy'))

            # Code for testing correctness
            # activation = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
            # top_K = 2

            for index in range(activation.shape[0]):
                max_index = activation[index, :].argsort()[-top_K:]
                activation[index, :] = np.zeros(activation.shape[1], np.float)
                for max in max_index:
                    activation[index, max] = 1.0
                if index % 1000 == 0:
                    print("Processing " + str(index) + "-th image")

            # print(activation)
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
        np.save(os.path.join(self.root_dir, 'convariance_' + str(top_K)), cov)
        mat_sum = np.sum(cov)
        print("Sum of elements is " + str(mat_sum) + " in a matrix with " + str(mat_sum.size) + " elements")


if __name__ == '__main__':
    converter = Converter(root_dir = '/home/ubuntu/sdf/activations/',
                          layer_list = ['conv3', 'conv4', 'conv5', 'fc6', 'fc7'])
    converter.output_binary(10)
    converter.output_covariance(10)