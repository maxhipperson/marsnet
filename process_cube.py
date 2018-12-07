import matplotlib
matplotlib.use('Qt5Agg')
import os
import hyperspy.api as hs
from skimage import io
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fitsne

from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import hdbscan

class Cube(object):
    def __init__(self):

        dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

        # river basin
        hdr_name = 'frt00003bfb_07_if166j_mtr3.hdr'
        cube_name = 'frt00003bfb_data_cube.tif'

        # # crater
        # hdr_name = 'frt00009a16_07_if166j_mtr3.hdr'
        # cube_name = 'frt00009a16_data_cube.tif'

        hdr_path = os.path.join(dir, hdr_name)
        cube_path = os.path.join(dir, cube_name)

        self.read_data_cube(hdr_path, cube_path)
        self.crop_signal(lower=1000, upper=2800)

        self.im.plot()
        plt.show()

        # crop image for clustering
        # self.im = self.im.inav[310:330, 470:490]  # 20 x 20
        # self.im = self.im.inav[290:350, 450:510]  # 60 x 60
        # self.im = self.im.inav[270:370, 430:530]  # 100 x 100
        # self.im = self.im.inav[250:390, 410:550]  # 140 x 140
        # self.im = self.im.inav[220:420, 380:580]  # 200 x 200
        # self.im = self.im.inav[180:480, 330:630] #
        # self.im = self.im.inav[120:650, 120:650]  #

        # set centre of image
        im_centre = (400, 400)

        # set size of image in pixels
        # size = 500
        # size = 400
        # size = 300
        # size = 200
        size = 100
        # size = 50
        # size = 20
        # size = 10

        self.im = self.im.inav[im_centre[0]-size/2:im_centre[0]+size/2, im_centre[1]-size/2:im_centre[1]+size/2]

        # maximum crop of image
        # self.im = self.im.inav[110:690, 120:760]  # 580 x 640 - centre = (400. 440)
        # self.im = self.im.inav[150:650, 190:690]  # 500 x 500
        # self.im = self.im.inav[200:600, 240:640]  # 400 x 400
        # self.im = self.im.inav[250:550, 290:590]  # 300 x 300
        # self.im = self.im.inav[300:500, 340:540]  # 200 x 200
        # self.im = self.im.inav[350:450, 390:490]  # 100 x 100
        # self.im = self.im.inav[375:425, 415:465]  # 50 x 50
        # self.im = self.im.inav[390:410, 430:450]  # 20 x 20
        # self.im = self.im.inav[395:405, 435:445]  # 10 x 10


        self.im.plot()
        plt.show()

        self.Z = self.im.data
        self.y, self.x, self.ch = self.Z.shape

        #################
        # Decomposition #
        #################

        # n_components = [0, 1, 2, 4, 5, 6, 7]  # 'frt00009a16_data_cube.tif'
        # n_components = 9  # 'frt00009a16_data_cube.tif'
        # n_components = 12

        # self.run_pca()
        # self.run_ica(n_components)
        # self.build_signal_from_decomposition(n_components)

        ##############
        # Clustering #
        ##############

        self.preprocess_clustering()

        #####################
        # Scikit clustering #
        #####################

        # DBSCAN


        ################
        # Pyclustering #
        ################

        # n_clusters = self.run_elbow()
        # n_cluststers = 3
        n_clusters = 4

        label_arr = self.k_means(n_clusters)
        self.plot_mean_and_mask(label_arr, 'K-Means, n_clusters: {}'.format(n_clusters))

        # n_initial = 8
        # n_max = 20
        # label_arr = self.x_means(n_initial, n_max)
        # self.plot_mean_and_mask(label_arr, 'X-means')

        ############
        # HBDBSCAN #
        ############

        # label_arr = self.hdbscan()
        # self.plot_mean_and_mask(label_arr, 'HBDBSCAN')

        ##########
        # FITSNE #
        ##########

        # self.run_tsne()
        self.run_tsne(labels=np.reshape(label_arr.copy(), (self.x * self.y)))

    @timeit
    def read_data_cube(self, hdr_path, img_path):
        """
        Reads a .tif data cube and a header file in from the paths specified.

        :param hdr_path:
        :param img_path:
        :return:
        """

        hdr = read_hdr_file(hdr_path)
        im = io.imread(img_path)

        print('Loaded {}:'.format(hdr_path))
        print('Loaded {}'.format(img_path))

        c, w, h = im.shape
        im = im.transpose(1, 2, 0)

        # remove ignored values
        im[im == hdr['data ignore value']] = 0

        (y, x, ch) = im.shape

        axes_x = {'name': 'x', 'size': x, 'units': 'px'}
        axes_y = {'name': 'y', 'size': y, 'units': 'px'}
        axes_ch = {'name': 'wavelength band', 'size': ch, 'units': 'index'}

        # convert image to signal object
        im = hs.signals.Signal1D(im, axes=[axes_y, axes_x, axes_ch])
        print(im.axes_manager)

        self.im = im
        self.hdr = hdr

    def crop_signal(self, lower, upper):
        """ Crops the signal to a wavelength range by indexing from the header file. """

        # find the index of the boundary wavelengths in the header
        wavelength = np.array(self.hdr['wavelength'])
        lower_index = np.argmax(wavelength >= lower)
        upper_index = np.argmax(wavelength > upper) - 1

        # crop the signal to the index range from header
        self.im.crop_signal1D(lower_index, upper_index)

        print('Cropped signal to {} - {} {}'.format(lower, upper, self.hdr['wavelength units']))
        print(self.im.axes_manager)

    @timeit
    def run_pca(self):

        self.im.decomposition()
        self.im.learning_results.summary()

        self.im.plot_explained_variance_ratio(threshold=0.00001)
        plt.show()

        self.im.plot_decomposition_results()
        plt.show()

    @timeit
    def run_ica(self, n_components):

        self.im.blind_source_separation(number_of_components=n_components)
        self.im.learning_results.summary()

        self.im.plot_bss_results()
        plt.show()

    @timeit
    def build_signal_from_decomposition(self, n_components):
        """ Builds a model of the signal from specified components - integer or list of integers. """

        self.signal_model = self.im.get_decomposition_model(n_components)

        self.signal_model.plot()
        plt.show()

        self.Z = self.signal_model.data

    @staticmethod
    def rescale_array(arr, axis=1):
        """ Mean subtracts and divides by the standard deviation spectrum wise. """

        arr -= arr.mean(axis=axis, keepdims=True)
        arr /= arr.std(axis=axis, keepdims=True)

        return arr

    @staticmethod
    def plot_mean(arr):
        """  """

        arr = np.sum(arr, axis=2)
        arr -= np.mean(arr)
        arr /= np.std(arr)

        return arr

    def plot_mean_and_mask(self, label_arr, label_arr_title='Labels'):
        """ Plots the mean image from 3D array and the mean image with an overlay. """

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        # fig.suptitle('')

        axes[0].set_title('Mean')
        axes[0].y_label('pixels / px')
        axes[0].x_label('pixels / px')
        axes[0].imshow(self.plot_mean(self.Z))

        axes[1].set_title(label_arr_title)
        axes[1].y_label('pixels / px')
        axes[1].x_label('pixels / px')
        axes[1].imshow(self.plot_mean(self.Z))
        axes[1].imshow(label_arr, cmap='Set1', alpha=0.4)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.95)
        plt.show()

    def preprocess_clustering(self):

        # flatten data and rescale spectra
        cluster_arr = np.reshape(self.Z.copy(), (self.x * self.y, self.ch))
        self.cluster_arr = self.rescale_array(cluster_arr)

        print('Flattened array shape: {}'.format(self.cluster_arr.shape))

    def rearrange_clusters_into_label_arr(self, clusters):

        # Only needed to rearrange the pyclustering label outut!

        print('Number of clusters: {}'.format(len(clusters)))

        label_arr = np.zeros((self.cluster_arr.shape[0]))

        for cluster_id, list in enumerate(clusters):
            for index in list:
                label_arr[index] = cluster_id

        print('Flattened label array shape: {}'.format(label_arr.shape))

        label_arr = label_arr.reshape((self.y, self.x))

        return label_arr

    def run_silhouette(self):
        raise NotImplementedError

    @timeit
    def run_elbow(self):

        # set k range
        kmin = 1
        kmax = 15

        elbow_instance = elbow(self.cluster_arr, kmin, kmax, ccore=False)
        elbow_instance.process()

        amount_clusters = elbow_instance.get_amount()  # most probable amount of clusters
        print('Amount of clusters: {}'.format(amount_clusters))

        wce = elbow_instance.get_wce()
        x = range(kmin, kmax)

        plt.figure(figsize=(5, 5))
        plt.title('Elbow plot: {} - {} clusters'.format(kmin, kmax))
        plt.ylabel('wce')
        plt.xlabel('k')
        plt.plot(x, wce, marker='x', linewidth=1)
        plt.plot(amount_clusters, wce[amount_clusters-1], marker='x', c='r')
        plt.tight_layout()
        plt.show()

        return amount_clusters

    @timeit
    def k_means(self,  n_clusters):

        # initialize initial centers using K-Means++ method
        initial_centers = kmeans_plusplus_initializer(self.cluster_arr, n_clusters).initialize()

        # create instance of K-Means algorithm with prepared centers
        kmeans_instance = kmeans(self.cluster_arr, initial_centers, ccore=False)
        kmeans_instance.process()

        clusters = kmeans_instance.get_clusters()
        label_arr = self.rearrange_clusters_into_label_arr(clusters)

        return label_arr

    @timeit
    def x_means(self, n_initial, n_max):

        initial_centers = kmeans_plusplus_initializer(self.cluster_arr, n_initial).initialize()

        # Create instance of X-Means algorithm.
        xmeans_instance = xmeans(self.cluster_arr, initial_centers, kmax=n_max, ccore=False)
        xmeans_instance.process()

        clusters = xmeans_instance.get_clusters()
        label_arr = self.rearrange_clusters_into_label_arr(clusters)

        return label_arr

    def run_tsne(self, labels=None):
        # todo add probabilities to cluster alpha

        print('Running fitsne...')

        Z = fitsne.FItSNE(self.cluster_arr.astype(np.double))

        plt.figure(figsize=(5, 5))
        if labels is not None:
            plt.scatter(Z[:, 0], Z[:, 1], s=1, alpha=0.3, c=labels, cmap='Set1')
        else:
            plt.scatter(Z[:, 0], Z[:, 1], s=1, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def DBSCAN(self):
        raise NotImplementedError

    @timeit
    def hdbscan(self):

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,  # primary parameter
            min_samples=2,
            metric='euclidean',
            core_dist_n_jobs=-1
        )

        clusterer.fit(self.cluster_arr)

        clusters = clusterer.labels_
        label_arr = clusters.reshape((self.y, self.x))

        print('Done')
        return label_arr





if __name__ == '__main__':

    cube = Cube()
