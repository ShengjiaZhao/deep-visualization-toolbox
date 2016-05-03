__author__ = 'shengjia'


import numpy as np
import os


if __name__ == '__main__':
    top_K = 10
    root_dir = '/home/ubuntu/sdf/activations/'
    layer_list = ['conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    for layer in layer_list:
        print("Transforming layer " + layer)
        activation = np.load(os.path.join(root_dir, layer + '.npy'))

        # Code for testing correctness
        # activation = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
        # top_K = 2

        for index in range(activation.shape[0]):
            max_index = activation[index, :].argsort()[-top_K:]
            activation[index, :] = np.zeros(activation.shape[1], np.float)
            for max in max_index:
                activation[index, max] = 1.0
            if index % 100 == 0:
                print("Processing " + str(index))

        # print(activation)
        np.save(os.path.join(root_dir, 'binary_' + layer))
