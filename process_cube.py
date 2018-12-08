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

        src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

        # river basin
        hdr_name = 'frt00003bfb_07_if166j_mtr3.hdr'
        im_name = 'frt00003bfb_data_cube.tif'
        x, y = (110, 690), (120, 760)  # 580 x 640 - centre = (400. 440)

        # # crater
        # hdr_name = 'frt00009a16_07_if166j_mtr3.hdr'
        # cube_name = 'frt00009a16_data_cube.tif'

        hdr_path = os.path.join(src_dir, hdr_name)
        im_path = os.path.join(src_dir, im_name)

        self.hdr = read_hdr_file(hdr_path)
        self.im = self.read_cube(im_path)

        print('Loaded {}:'.format(hdr_path))
        print('Loaded {}'.format(im_path))

        # self.plot(self.im)

        # Restrict wavelength range
        # lower, upper = 1000, 2800
        # lower, upper = 1000, 2800
        lower, upper = None, 2800
        # lower, upper = 1000, None

        # Crop signal to wavelength or index range
        self.crop_signal(lower=lower, upper=upper)
        # self.plot(self.im)

        # 'frt00009a16_data_cube.tif'
        # Set size of image to crop to (in pixels)
        self.im = self.im.inav[x[0]:x[1], y[0]:y[1]]

        print(self.im.axes_manager)
        # self.plot(self.im)  # todo disable temporarily

        #######################################################
        # Decomposition

        # self.n_components_ica = None
        self.n_components_ica = 14

        # self.n_components_model = None
        # self.n_components_model = 14
        self.n_components_model = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]  # todo set from pca / ica

        self.signal_model = None

        # plot = True
        plot = False

        # self.run_pca(plot)
        # self.run_ica(plot)
        # self.signal_model = self.build_signal_from_decomposition()

        #######################################################
        # Crop image

        # Set signal for analyis
        if self.signal_model is not None:
            self.signal = self.signal_model.copy()
        else:
            self.signal = self.im.copy()

        # Set size of image to crop to (in pixels)  # todo set
        new_size = 10
        # new_size = 20
        # new_size = 50
        # new_size = 100
        # new_size = 200
        # new_size = 300
        # new_size = 400
        # new_size = 500

        x_mid = (x[0] + x[1]) / 2
        y_mid = (y[0] + y[1]) / 2
        x_min = x_mid - new_size / 2
        x_max = x_mid + new_size / 2
        y_min = y_mid - new_size / 2
        y_max = y_mid + new_size / 2

        self.signal = self.signal.inav[x_min:x_max, y_min:y_max]
        print(self.signal.axes_manager)
        # self.plot(self.signal)  # todo disable temporarily

        #######################################################
        # Set data

        self.Z = self.signal.data
        self.y, self.x, self.ch = self.Z.shape

        #######################################################
        # Clustering

        self.label_arr = None
        self.cluster_arr = self.preprocess_clustering()

        #######################################################
        # K-Means

        savefig = True
        noshow = True  # todo enable temporarily

        savedir = '/Users/maxhipperson/Documents/Year 4/marsnet/results/frt00003bfb/no_decomposition'
        # self.savedir = '/Users/maxhipperson/Documents/Year 4/marsnet/results/frt00003bfb/pca_model'
        # self.savedir = '/Users/maxhipperson/Documents/Year 4/marsnet/results/frt00003bfb/ica_model'

        n_clusters = self.plot_elbow(noshow=noshow,
                                     save=True,
                                     savepath=os.path.join(savedir,'elbow.imsize_{}.png'.format(new_size)),)
        # n_clusters = 4
        # n_clusters = 8
        n_clusters = [2, 3, 4, 6, 8]

        for n in n_clusters:
            self.label_arr = self.k_means(n)
            self.plot_mean_and_mask(label_arr_title='K-Means, n_clusters: {}'.format(n),
                                    noshow=noshow,
                                    save=savefig,
                                    savepath=os.path.join(savedir,
                                                          'kmeans.imsize_{}.nclusters_{}.png'.format(new_size, n)))

        # n_initial = 8
        # n_max = 20
        # label_arr = self.x_means(n_initial, n_max)
        # self.plot_mean_and_mask(label_arr, 'X-means')

        #######################################################
        # HBDBSCAN

        # label_arr = self.hdbscan()
        # self.plot_mean_and_mask(label_arr, 'HBDBSCAN')

        #######################################################
        # FITSNE

        # self.tsne()

    def read_cube(self, im_path):
        """ Reads a .tif data cube and a header file in from the paths specified. """

        im = io.imread(im_path)
        im = im.transpose(1, 2, 0)
        (y, x, ch) = im.shape

        # remove ignored values
        im[im == self.hdr['data ignore value']] = 0

        # set the hyperspy axis parameters
        axes_x = {'name': 'x', 'size': x, 'units': 'px'}
        axes_y = {'name': 'y', 'size': y, 'units': 'px'}
        axes_ch = {'name': 'wavelength band', 'size': ch, 'units': 'index'}

        # convert image to signal object
        im = hs.signals.Signal1D(im, axes=[axes_y, axes_x, axes_ch])

        return im

    @staticmethod
    def plot(im):
        im.plot()
        plt.show()

    def crop_signal(self, lower, upper, limit_type='wavelength'):
        """ Crops the signal to a wavelength range by indexing from the header file. """

        # find the index of the boundary wavelengths in the header
        if limit_type == 'wavelength':
            wavelength = np.array(self.hdr['wavelength'])

            if lower is None:
                lower_index = None
            else:
                lower_index = np.argmax(wavelength >= lower)

            if upper is None:
                upper_index = None
            else:
                upper_index = np.argmax(wavelength > upper) - 1

        else:
            lower_index = lower
            upper_index = upper

        # crop the signal to the index range from header
        self.im.crop_signal1D(lower_index, upper_index)

        print('Cropped signal')
        print('wavelength: {} - {}'.format(lower, upper, self.hdr['wavelength units']))
        print('index: {} - {}'.format(lower_index, upper_index))

    def run_pca(self, plot=True):

        self.im.decomposition()
        self.im.learning_results.summary()

        if plot:
            self.im.plot_explained_variance_ratio(threshold=0.00001)
            plt.show()
            self.im.plot_decomposition_results()
            plt.show()

    def run_ica(self, plot=True):

        self.im.blind_source_separation(self.n_components_ica)
        self.im.learning_results.summary()

        if plot:
            self.im.plot_bss_results()
            plt.show()

    def build_signal_from_decomposition(self):
        """ Builds a model of the signal from specified components - integer or list of integers. """

        signal_model = self.im.get_decomposition_model(self.n_components_model)
        signal_model.plot()
        plt.show()

        return signal_model

    @staticmethod
    def rescale_array(arr, axis=1):
        """ Mean subtracts and divides by the standard deviation spectrum wise. """

        arr -= arr.mean(axis=axis, keepdims=True)
        arr /= arr.std(axis=axis, keepdims=True)

        return arr

    def plot_mean_and_mask(self, data=None, label_arr=None, label_arr_title='Labels', save=False, savepath='fig.png', noshow=False):
        """ Plots the mean image from 3D array and the mean image with an overlay. """

        if data is None:
            data = self.Z.copy()

        if label_arr is None:
            label_arr = self.label_arr.copy()

        data = np.sum(data, axis=2)
        data -= np.mean(data)
        data /= np.std(data)

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        # fig.suptitle('')

        axes[0].set_title('Mean')
        axes[0].set_ylabel('pixels / px')
        axes[0].set_xlabel('pixels / px')
        axes[0].imshow(data)

        # Reset matplotlib colour cycle
        axes[1].set_prop_cycle(None)
        axes[1].set_title(label_arr_title)
        axes[1].set_ylabel('pixels / px')
        axes[1].set_xlabel('pixels / px')
        axes[1].imshow(data)
        axes[1].imshow(label_arr, cmap='Set1', alpha=0.4)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.95)

        if save:
            fig.savefig(savepath)
            if noshow:
                plt.close()

        plt.show()

    def preprocess_clustering(self):

        # flatten data and rescale spectra
        cluster_arr = np.reshape(self.Z.copy(), (self.x * self.y, self.ch))
        cluster_arr = self.rescale_array(cluster_arr)

        print('Flattened array shape: {}'.format(cluster_arr.shape))

        return cluster_arr


    def rearrange_clusters_into_label_arr(self, clusters):

        # Only needed to rearrange the pyclustering label outut!
        print('Number of clusters: {}'.format(len(clusters)))

        label_arr = np.zeros((self.cluster_arr.shape[0]))

        for cluster_id, spectra_list in enumerate(clusters):
            for index in spectra_list:
                label_arr[index] = cluster_id

        label_arr = label_arr.reshape((self.y, self.x))

        return label_arr


    @timeit
    def k_means(self,  n_clusters=None):

        data = self.cluster_arr

        # initialize centers
        initial_centers = kmeans_plusplus_initializer(data, n_clusters).initialize()

        # create instance of K-Means with centers
        kmeans_instance = kmeans(data, initial_centers)
        kmeans_instance.process()

        # get clusters and process into label_arr
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()

        label_arr = self.rearrange_clusters_into_label_arr(clusters)

        # temp

        # spectra = [[self.cluster_arr[index] for index in spectra_list] for spectra_list in clusters]
        # spectra_arr = [np.array(f) for f in spectra]
        # mean_spectra = [np.mean(f, axis=0) for f in spectra_arr]
        #
        # plt.figure()
        # for spectrum in mean_spectra:
        #     plt.plot(range(spectrum.size), spectrum)
        # plt.show()

        return label_arr


    @timeit
    def plot_elbow(self, save=False, savepath='elbow.png', noshow=False):
        # set k range
        kmin = 1
        kmax = 15

        elbow_instance = elbow(self.cluster_arr, kmin, kmax)
        elbow_instance.process()

        amount_clusters = elbow_instance.get_amount()  # most probable amount of clusters
        print('Amount of clusters: {}'.format(amount_clusters))

        wce = elbow_instance.get_wce()
        x = range(kmin, kmax)

        plt.figure(figsize=(5, 5))
        plt.title('Elbow plot, {} - {} clusters'.format(kmin, kmax))
        plt.ylabel('wce')
        plt.xlabel('k')
        plt.plot(x, wce, marker='x', linewidth=1)
        plt.plot(amount_clusters, wce[amount_clusters - 1], marker='x', c='r')
        plt.tight_layout()

        if save:
            plt.savefig(savepath)
            if noshow:
                plt.close()

        plt.show()

        return amount_clusters

    @timeit
    def tsne(self, data=None, label_arr=None):

        if data is None:
            data = self.cluster_arr.astype(np.double)

        if self.cluster_arr is not None:
            label_arr = np.reshape(self.label_arr.copy(), (self.x * self.y))

        out = fitsne.FItSNE(data)

        plt.figure(figsize=(10, 10))
        plt.scatter(out[:, 0], out[:, 1], s=1, alpha=0.3, c=label_arr, cmap='Set1')
        plt.tight_layout()
        plt.show()


    @timeit
    def hdbscan(self):

        clusterer = hdbscan.HDBSCAN(min_cluster_size=3,
                                    min_samples=2,
                                    metric='euclidean',
                                    core_dist_n_jobs=-1)

        clusterer.fit(self.cluster_arr)

        clusters = clusterer.labels_
        label_arr = clusters.reshape((self.y, self.x))

        print('Done')
        return label_arr





if __name__ == '__main__':

    cube = Cube()
