__author__ = 'shengjia'

from config import settings
import numpy as np


class ImageSearcher:
    def __init__(self):
        self.activation = np.load(settings['activation_root'] + 'fc6.npy')
        self.num_image = self.activation.shape[0]

        # Read in path for all the images
        image_list_file = open(settings['activation_root'] + 'input.txt')
        self.image_list = []
        while True:
            path = image_list_file.readline()
            if not path:
                break
            self.image_list.append(path.strip())

    def query(self, query, num_hits=10):
        diff_matrix = self.activation - np.tile(self.activation[query, :], (self.num_image, 1))
        order = np.argsort(np.sum(abs(diff_matrix), 1))
        result = []
        for i in range(0, num_hits):
            result.append(settings['image_root'] + self.image_list[order[i]])
        return result

    def get_path(self, index):
        return settings['image_root'] + self.image_list[index]