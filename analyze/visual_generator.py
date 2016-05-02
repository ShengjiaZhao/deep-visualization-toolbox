__author__ = 'shengjia'

import numpy as np
from matplotlib import pyplot as plt
import os, random, time, math
from scipy import misc


class VisualGenerator:
    def __init__(self, image_folder, K=9):
        self.image_folder = image_folder
        self.K = 9
        self.slot_count = 50
        self.display_size = [1080, 1920]

    def visualize_collage(self, nodes, include_deconv=False):
        image_width = (self.display_size[0] + self.display_size[1]) / math.sqrt(len(nodes))
        if image_width > min(self.display_size) / 2:
            image_width = min(self.display_size) / 2
        if image_width < 100:
            image_width = 100
        image_width = int(image_width)
        print(image_width)
        canvas = np.zeros(self.display_size + [3], np.uint8)
        for node in nodes:
            if include_deconv:
                plot_x = int(math.floor(node['coord'][0] * (self.display_size[1] - image_width * 2)))
            else:
                plot_x = int(math.floor(node['coord'][0] * (self.display_size[1] - image_width)))
            plot_y = int(math.floor(node['coord'][1] * (self.display_size[0] - image_width)))

            max_path = os.path.join(self.image_folder, node['layer'],
                                    'unit_%.4d' % node['index'], 'maxim_000.png')
            if not os.path.isfile(max_path):
                print("Error: " + max_path + " do not exist")
            else:
                canvas[plot_y:plot_y+image_width,
                        plot_x:plot_x+image_width, :] = \
                        misc.imresize(misc.imread(max_path), (image_width, image_width))

            if include_deconv:
                deconv_path = os.path.join(self.image_folder, node['layer'],
                           'unit_%.4d' % node['index'], 'deconv_000.png')
                if not os.path.isfile(deconv_path):
                    print("Error: " + deconv_path + " do not exist")
                else:
                    canvas[plot_y:plot_y+image_width,
                            plot_x+image_width:plot_x+image_width*2, :] = \
                            misc.imresize(misc.imread(deconv_path), (image_width, image_width))

            if include_deconv:
                border_width = image_width*2
            else:
                border_width = image_width
            thickness = 2
            canvas[plot_y:plot_y+image_width, plot_x:plot_x+thickness, :] = \
                np.zeros((image_width, thickness, 3), np.uint8)
            canvas[plot_y:plot_y+image_width, plot_x+border_width-thickness:plot_x+border_width, :] = \
                np.zeros((image_width, thickness, 3), np.uint8)
            canvas[plot_y:plot_y+thickness, plot_x:plot_x+border_width, :] = \
                np.zeros((thickness, border_width, 3), np.uint8)
            canvas[plot_y+image_width-thickness:plot_y+image_width, plot_x:plot_x+border_width, :] = \
                np.zeros((thickness, border_width, 3), np.uint8)
        plt.cla()
        plt.imshow(canvas)
        plt.show()
        time.sleep(0.2)

    def visualize_in_grid(self, nodes):
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
    for i in range(1, 100):
        show_list.append({'layer': 'conv4', 'index': i, 'coord': [random.random(), random.random()]})
    #gen.visualize_in_grid(show_list)
    gen.visualize_collage(show_list)
