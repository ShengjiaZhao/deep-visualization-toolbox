__author__ = 'shengjia'

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import math
# generate two clusters: a with 100 points, b with 50:
import random, os
from visual_generator import VisualGenerator
#np.random.seed(4711)  # for repeatability of this tutorial

output_path = 'cluster_output/'


class ImageClustering:
    """ Backend class for computing the image clusters
    """
    def __init__(self, settings=None, layers=['conv5'], num_samples=5000):
        self.ddata = None

        self.visualizer = VisualGenerator(settings)
        # Read in list of image files
        image_list_file = open(settings['activation_root'] + 'input.txt')
        self.image_list = []
        while True:
            path = image_list_file.readline()
            if not path:
                break
            self.image_list.append(path.strip())

        # Read in activation files
        activation_list = []
        for layer in layers:
            activation_list.append(np.load(settings['activation_root'] + layer + '.npy'))
        activation = np.concatenate(activation_list, 0)

        # Normalize
        min_mat = np.tile(np.expand_dims(np.min(activation, 0), 0), (activation.shape[0], 1))
        range_mat = np.max(activation, 0) - np.min(activation, 0)
        range_mat = np.clip(range_mat, 1e-6, np.max(range_mat))
        range_mat = np.tile(np.expand_dims(range_mat, 0), (activation.shape[0], 1))
        activation = np.divide(np.subtract(activation, min_mat), range_mat)
        activation -= 0.2
        activation = np.clip(activation, 0, 1.0)

        # Select some random images
        if num_samples is not None:
            new_activation = np.ndarray((num_samples, activation.shape[1]))
            new_image_list = []
            indexes = random.sample(range(len(self.image_list)), num_samples)
            for i in range(num_samples):
                new_activation[i, :] = activation[indexes[i], :]
                new_image_list.append(self.image_list[indexes[i]])
            self.image_list = new_image_list
            self.activation = new_activation
        else:
            self.activation = activation

    def generate_synthetic(self):
        num_normals = 10
        normals = []
        for i in range(num_normals):
            cov = np.random.random((2, 2))
            cov = (cov * cov.transpose() + 10*np.diag(np.random.random(2))) / 200
            normals.append(np.random.multivariate_normal(np.random.random(2), cov, size=[50,]))
        x = np.concatenate(normals)
        return x

    def cluster(self, data=None):
        if data is None:
            data = self.activation
        data_count = data.shape[0]
        # generate the linkage matrix
        Z = linkage(data, 'ward')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        np.save(output_path + 'linkage', Z)
        print("Computed linkage")

        # Create visualizations for each large cluster
        for index in range(1, 100):
            images_in_cluster = []
            index_stack = [data_count * 2 - index - 1]
            while index_stack:
                cur_index = index_stack.pop()
                if cur_index < data_count:
                    images_in_cluster.append(cur_index)
                else:
                    index_stack.append(int(round(Z[cur_index - data_count, 0])))
                    index_stack.append(int(round(Z[cur_index - data_count, 1])))

            # Sample images if there are less
            if len(images_in_cluster) > 100:
                images_in_cluster = random.sample(images_in_cluster, 100)
            images = [{'path': self.image_list[image_index], 'coord': [random.random(), random.random()]}
                      for image_index in images_in_cluster]
            # fig = plt.figure(figsize=(25, 10))
            img = self.visualizer.visualize_collage_image(images)
            np.save(output_path + 'cluster' + str(index), img)
            print("Processed img " + str(index))

    def plot(self, data):
        plt.scatter(data[:,0], data[:,1])
        plt.show()


class ImageClusterDisplay:
    def __init__(self):
        self.ax = None
        self.ddata = None
        self.x_range = None
        self.y_range = None
        self.mapping = None

        self.text_list = {
            0: 'all',
            4: 'non-living',
            1: 'living or related',
            11: 'quadruped',
            6: 'natural',
            9: 'human related',
            21: 'people',
            45: 'dogs',
            52: 'insect',
            13: 'water life',
            19: 'appliances',
            16: 'indoor scenes',
            8: 'outdoor scenes',
            39: 'can fly',
            12: 'can\'t fly',
        }
    def display(self):
        self.linkage = np.load(output_path + 'linkage.npy')
        # Plot dendrogram
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendro_p = 100
        self.ddata = dendrogram(
            self.linkage,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            truncate_mode='lastp',  # show only the last p merged clusters
            p=dendro_p,  # show only the last p merged clusters
            show_leaf_counts=True,
        )
        ordering = np.argsort([d[1] for d in self.ddata['dcoord']])[::-1]
        # Map each item in linkage to its cluster index
        self.mapping = np.ndarray(dendro_p-1, np.int)
        for index in range(dendro_p - 1):
            self.mapping[ordering[index]] = index

        for i, d, c, index in zip(self.ddata['icoord'], self.ddata['dcoord'], self.ddata['color_list'], range(dendro_p-1)):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > 2:
                plt.plot(x, y, 'o', c=c)
                plt.annotate(str(self.mapping[index]), (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
                if self.mapping[index] in self.text_list:
                    plt.annotate(self.text_list[self.mapping[index]], (x, y), xytext=(0, 2), textcoords='offset points',
                                 va='bottom', ha='center')

        self.x_range = np.max(self.ddata['icoord']) - np.min(self.ddata['icoord'])
        self.y_range = np.max(self.ddata['dcoord']) - np.min(self.ddata['dcoord'])
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

    # Event Handler
    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        for i, d, index in zip(self.ddata['icoord'], self.ddata['dcoord'], xrange(self.linkage.shape[0])):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if abs(event.xdata - x) / self.x_range < 0.01 and abs(event.ydata - y) / self.y_range < 0.01:
                # circ = Ellipse(xy=(x, y), width=10, height=0.1)
                # self.ax.add_artist(circ)
                # plt.draw()
                img = np.load(output_path + 'cluster' + str(self.mapping[index]+1) + '.npy')
                plt.figure(figsize=(20, 20))
                plt.imshow(img)
                plt.title('Cluster ' + str(self.mapping[index]))
                plt.show()
                break

if __name__ == '__main__':
    # clusterer = ImageClustering()
    # x = clusterer.generate_synthetic()
    #clusterer.cluster(x)
    displayer = ImageClusterDisplay()
    displayer.display()