__author__ = 'shengjia'

import math
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from scipy import misc


class VisualGenerator:
    def __init__(self, image_folder, K=9):
        self.image_folder = image_folder
        self.K = 9
        self.slot_count = 50
        self.display_size = [1080, 1920]

    def visualize(self, nodes):
        activation_per_node = int(self.slot_count / len(nodes))
        if activation_per_node > self.K:
            activation_per_node = self.K
        if activation_per_node < 1:
            activation_per_node = 1
        total_slots = len(nodes) * activation_per_node
        ratio = float(self.display_size[1]) / self.display_size[0]
        column_count = int(math.ceil(math.sqrt(total_slots * ratio / 2)))
        row_count = int(math.ceil(float(total_slots) / column_count))
        print(activation_per_node, total_slots, column_count * 2, row_count, row_count * column_count)

        canvas = np.zeros(self.display_size + [3], np.uint8)
        image_width = int(math.floor(float(self.display_size[1]) / column_count / 2))
        image_height = int(math.floor(float(self.display_size[0]) / row_count))
        if image_width > image_height:
            image_width = image_height

        if row_count * column_count < total_slots:
            print("Error!")
        if image_width > image_height:
            print("Error!")
        slot_counter = 0
        for node in nodes:
            for index in range(activation_per_node):
                row = int(math.floor(slot_counter / column_count))
                col = slot_counter % column_count
                slot_counter += 1
                max_path = os.path.join(self.image_folder, node['layer'],
                                        'unit_%.4d' % node['index'], 'maxim_%.3d.png' % index)
                deconv_path = os.path.join(self.image_folder, node['layer'],
                                           'unit_%.4d' % node['index'], 'deconv_%.3d.png' % index)
                if not os.path.isfile(max_path):
                    print("Error: " + max_path + " do not exist")
                else:
                    canvas[row*image_width:row*image_width+image_width,
                            col*image_width*2:col*image_width*2+image_width, :] = \
                            misc.imresize(misc.imread(max_path), (image_width, image_width))
                if not os.path.isfile(deconv_path):
                    print("Error: " + deconv_path + " do not exist")
                else:
                    canvas[row*image_width:row*image_width+image_width,
                            col*image_width*2+image_width:(col+1)*image_width*2, :] = \
                            misc.imresize(misc.imread(deconv_path), (image_width, image_width))

        plt.cla()
        plt.imshow(canvas)
        plt.show()
        time.sleep(0.2)


if __name__ == '__main__':
    gen = VisualGenerator('/home/shengjia/DeepLearning/deep-visualization-toolbox/result_out')
    show_list = []
    for i in range(50, 100):
        show_list.append({'layer': 'conv4', 'index': i})
    gen.visualize(show_list)

