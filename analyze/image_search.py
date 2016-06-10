__author__ = 'shengjia'


import numpy as np
sys.path.insert(0, '..')
from find_maxes.loaders import load_imagenet_mean, load_labels, caffe

class ImageSearcher:
    def __init__(self, net_prototxt, net_weights, settings):
        self.settings = settings
        self.layer = 'conv5'
        self.activation = np.load(settings['activation_root'] + self.layer + '.npy')[:80000]
        self.num_image = self.activation.shape[0]

        # Read in path for all the images
        image_list_file = open(settings['activation_root'] + 'input.txt')
        self.image_list = []
        while True:
            path = image_list_file.readline()
            if not path:
                break
            self.image_list.append(path.strip())

        imagenet_mean = load_imagenet_mean()
        self.net = caffe.Classifier(net_prototxt, net_weights,
                                    mean=imagenet_mean,
                                    channel_swap=(2,1,0),
                                    raw_scale=255,
                                    image_dims=(256, 256))

    def query(self, query, num_hits=10):
        diff_matrix = self.activation - np.tile(self.activation[query, :], (self.num_image, 1))
        order = np.argsort(np.sum(abs(diff_matrix), 1))
        result = []
        for i in range(0, num_hits):
            result.append(self.settings['image_root'] + self.image_list[order[i]])
        return result

    def get_path(self, index):
        return self.settings['image_root'] + self.image_list[index]

    def query_image(self, image_path, num_hits=10):
        im = caffe.io.load_image(image_path)
        self.net.predict([im], oversample=False)   # Just take center crop
        layer_shape = self.net.blobs[self.layer].data.shape
        if len(layer_shape) == 4:
            result_array = np.amax(self.net.blobs[self.layer].data, (0, 2, 3))
        else:
            result_array = np.amax(self.net.blobs[self.layer].data, 0)

        diff_matrix = self.activation - np.tile(result_array, (self.num_image, 1))
        order = np.argsort(np.sum(abs(diff_matrix), 1))
        result = []
        for i in range(0, num_hits):
            result.append(self.settings['image_root'] + self.image_list[order[i]])
        return result
