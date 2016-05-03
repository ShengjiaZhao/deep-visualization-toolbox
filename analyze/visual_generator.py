__author__ = 'shengjia'

import numpy as np
from matplotlib import pyplot as plt
import os, random, time, math
from scipy import misc


class VisualGenerator:
    def __init__(self, image_folder, node_folder, K=9):
        self.image_folder = image_folder
        self.node_folder = node_folder
        self.K = 9
        self.slot_count = 50
        self.display_size = [1080, 1920]
        self.sparsity_level = 2

    def get_image_width(self, count):
        image_width = (self.display_size[0] + self.display_size[1]) / math.sqrt(count) / self.sparsity_level
        if image_width > min(self.display_size) / 2:
            image_width = min(self.display_size) / 2
        if image_width < 100:
            image_width = 100
        return int(image_width)

    def draw_border(self, canvas, plot_x, plot_y, width, height, thickness):
        canvas[plot_y:plot_y+height, plot_x:plot_x+thickness, :] = \
            np.zeros((height, thickness, 3), np.uint8)
        canvas[plot_y:plot_y+height, plot_x+width-thickness:plot_x+width, :] = \
            np.zeros((height, thickness, 3), np.uint8)
        canvas[plot_y:plot_y+thickness, plot_x:plot_x+width, :] = \
            np.zeros((thickness, width, 3), np.uint8)
        canvas[plot_y+height-thickness:plot_y+height, plot_x:plot_x+width, :] = \
            np.zeros((thickness, width, 3), np.uint8)

    def normalize(self, array):
        x_coords = [obj['coord'][0] for obj in array]
        y_coords = [obj['coord'][1] for obj in array]
        x_max = np.max(x_coords)
        x_min = np.min(x_coords)
        y_max = np.max(y_coords)
        y_min = np.min(y_coords)
        for obj in array:
            obj['coord'][0] = (obj['coord'][0] - x_min) / (x_max - x_min)
            obj['coord'][1] = (obj['coord'][1] - y_min) / (y_max - y_min)

    def visualize_collage_image(self, images, keep_ratio=True):
        self.normalize(images)
        image_width = self.get_image_width(len(images))
        canvas = np.ones(self.display_size + [3], np.uint8) * 255
        for image in images:
            plot_x = int(math.floor(image['coord'][0] * (self.display_size[1] - image_width)))
            plot_y = int(math.floor(image['coord'][1] * (self.display_size[0] - image_width)))
            image_path = os.path.join(self.image_folder, image['path'])
            if not os.path.isfile(image_path):
                print("Error: " + image_path + " do not exist")
            else:
                img = misc.imread(image_path)
                if len(img.shape) == 2:
                    print("Warning: image " + image_path + " with only two channels")
                    img = np.tile(img, (1, 1, 3))
                    print(img.shape)
                if keep_ratio and img.shape[0] > img.shape[1]:
                    actual_height = image_width
                    actual_width = int(image_width * img.shape[1] / img.shape[0])
                elif keep_ratio and img.shape[0] < img.shape[1]:
                    actual_height = int(image_width * img.shape[0] / img.shape[1])
                    actual_width = image_width
                else:
                    actual_height = image_width
                    actual_width = image_width
                canvas[plot_y:plot_y+actual_height, plot_x:plot_x+actual_width, :] = \
                        misc.imresize(img, (actual_height, actual_width))
            self.draw_border(canvas, plot_x, plot_y, width=actual_width, height=actual_height, thickness=2)
        return canvas

    def visualize_collage_node(self, nodes, include_deconv=False):
        self.normalize(nodes)
        image_width = self.get_image_width(len(nodes))
        canvas = np.ones(self.display_size + [3], np.uint8) * 255
        for node in nodes:
            if include_deconv:
                plot_x = int(math.floor(node['coord'][0] * (self.display_size[1] - image_width * 2)))
            else:
                plot_x = int(math.floor(node['coord'][0] * (self.display_size[1] - image_width)))
            plot_y = int(math.floor(node['coord'][1] * (self.display_size[0] - image_width)))

            max_path = os.path.join(self.node_folder, node['layer'],
                                    'unit_%.4d' % node['index'], 'maxim_000.png')
            if not os.path.isfile(max_path):
                print("Error: " + max_path + " do not exist")
            else:
                canvas[plot_y:plot_y+image_width,
                        plot_x:plot_x+image_width, :] = \
                        misc.imresize(misc.imread(max_path), (image_width, image_width))

            if include_deconv:
                deconv_path = os.path.join(self.node_folder, node['layer'],
                           'unit_%.4d' % node['index'], 'deconv_000.png')
                if not os.path.isfile(deconv_path):
                    print("Error: " + deconv_path + " do not exist")
                else:
                    canvas[plot_y:plot_y+image_width,
                            plot_x+image_width:plot_x+image_width*2, :] = \
                            misc.imresize(misc.imread(deconv_path), (image_width, image_width))

            if include_deconv:
                self.draw_border(canvas, plot_x, plot_y, width=image_width*2, height=image_width, thickness=2)
            else:
                self.draw_border(canvas, plot_x, plot_y, width=image_width, height=image_width, thickness=2)
        return canvas

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
        return canvas


if __name__ == '__main__':
    gen = VisualGenerator(image_folder='/home/shengjia/deep-visualization-toolbox/input_images',
                          node_folder='/home/shengjia/DeepLearning/deep-visualization-toolbox/result_out')
    show_list = []
    for i in range(1, 100):
        show_list.append({'layer': 'conv4', 'index': i, 'coord': [random.random(), random.random()]})
    #gen.visualize_in_grid(show_list)
    #gen.visualize_collage_node(show_list, True)

    img_path_list = ['ILSVRC2012_val_00000610.jpg', 'ILSVRC2012_val_00006491.jpg'] * 10
    img_with_coord = [{'path': image_path, 'coord': [random.random(), random.random()]} for image_path in img_path_list]
    gen.visualize_collage_image(img_with_coord)