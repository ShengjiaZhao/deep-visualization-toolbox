__author__ = 'shengjia'

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import math
# generate two clusters: a with 100 points, b with 50:
import random
from visual_generator import VisualGenerator
#np.random.seed(4711)  # for repeatability of this tutorial


class ImageClustering:
    def __init__(self, settings=None, layers=['conv5']):
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
        num_samples = 5000
        new_activation = np.ndarray((num_samples, activation.shape[1]))
        new_image_list = []
        indexes = random.sample(range(len(self.image_list)), num_samples)
        for i in range(num_samples):
            new_activation[i, :] = activation[indexes[i], :]
            new_image_list.append(self.image_list[indexes[i]])
        self.image_list = new_image_list
        self.activation = new_activation

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
        print("Computed linkage")

        # Plot dendrogram
        fig = plt.figure(figsize=(25, 10))
        # self.ax = fig.add_subplot(1, 1, 1)
        # cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        self.ddata = dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            truncate_mode='lastp',  # show only the last p merged clusters
            p=200,  # show only the last p merged clusters
        )
        plt.show()

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
            print(len(images_in_cluster))
            # Sample images if there are less
            if len(images_in_cluster) > 100:
                images_in_cluster = random.sample(images_in_cluster, 100)
            images = [{'path': self.image_list[image_index], 'coord': [random.random(), random.random()]}
                      for image_index in images_in_cluster]
            plt.imshow(self.visualizer.visualize_collage_image(images))
            plt.show()
            ''' Plot and verify correctness
            point_list_x = [data[index, 0] for index in images_in_cluster]
            point_list_y = [data[index, 1] for index in images_in_cluster]
            plt.xlim([-1, 2])
            plt.ylim([-1, 2])
            plt.scatter(point_list_x, point_list_y)
            plt.show() '''
            #print(images_in_cluster)
            #

        self.x_range = np.max(self.ddata['icoord']) - np.min(self.ddata['icoord'])
        self.y_range = np.max(self.ddata['dcoord']) - np.min(self.ddata['dcoord'])


    def plot(self, data):
        plt.scatter(data[:,0], data[:,1])
        plt.show()

    # Event Handler
    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        for i, d in zip(self.ddata['icoord'], self.ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if abs(event.xdata - x) / self.x_range < 0.01 and abs(event.ydata - y) / self.y_range < 0.01:
                circ = Ellipse(xy=(x, y), width=10, height=0.1)
                self.ax.add_artist(circ)
                plt.draw()
                print("Hit: " + str(x) + " " + str(y))
                break

if __name__ == '__main__':
    clusterer = ImageClustering()
    x = clusterer.generate_synthetic()
    clusterer.cluster(x)