__author__ = 'shengjia'


import numpy as np
from visual_generator import VisualGenerator


class LSAAnalyzer:
    def __init__(self, image_list, node_list, visualizer):
        self.u = None
        self.s = None
        self.v = None
        self.image_list = image_list
        self.node_list = node_list
        self.visualizer = visualizer

    def svd(self, mat):
        assert mat.shape[0] == len(self.image_list) and mat.shape[1] == len(self.node_list)
        self.u, self.s, self.v = np.linalg.svd(mat, full_matrices=False)

    def visualize(self):
        if self.u is None or self.v is None:
            return
        image_feature = self.u[:, 0:2]
        unit_feature = self.v[0:2, :]

        img_min = np.min(image_feature)
        img_max = np.max(image_feature)
        image_feature = np.divide(image_feature - img_min, img_max - img_min)
        unit_min = np.min(unit_feature)
        unit_max = np.max(unit_feature)
        unit_feature = np.divide(unit_feature - unit_min, unit_max - unit_min)

        nodes = []
        for node, i in zip(self.node_list, range(len(self.node_list))):
            nodes.append({'layer': node['layer'], 'index': node['index'],
                          'coord': [unit_feature[1, i], unit_feature[0, i]]})
        images = []
        for image_path, i in zip(self.image_list, range(len(self.image_list))):
            images.append({'path': image_path,
                           'coord': [image_feature[i, 1], image_feature[i, 0]]})
        visualizer.visualize_collage_image(images)
        visualizer.visualize_collage_node(nodes)

    def plot_feature(self, feature):
        pass


if __name__ == '__main__':
    image_list_file = open('~/sdf/activation_images')
    image_list = []
    while True:
        path = image_list_file.readline()
        if path is None:
            break
        image_list.append(path)

    node_list = []
    for index in range(256):
        node_list.append({'layer': 'conv5', 'index': index})
    visualizer = VisualGenerator('~/sdf/images', '~/sdf/results')
    lsa = LSAAnalyzer(image_list, node_list, visualizer)
