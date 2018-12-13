import matplotlib
matplotlib.use('Qt5Agg')
from utils import *
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import hdbscan
import fitsne
from tqdm import tqdm
from sklearn.preprocessing import normalize


class Cube(object):
    def __init__(self, src_dir, hdr_name, im_name, wavelength_min=None, wavelength_max=None):

        hdr_path = os.path.join(src_dir, hdr_name)
        im_path = os.path.join(src_dir, im_name)

        self.hdr = read_hdr_file(hdr_path)
        self.im = self._read_cube(im_path)

        print('Loaded {}:'.format(hdr_path))
        print('Loaded {}'.format(im_path))

        self.signal = self.im.copy()

        self._crop_signal(lower=wavelength_min, upper=wavelength_max, limit_type='wavelength')
        self._set_data(self.signal)

        self.spectra_index_arr = None
        self.spectra_arr = None
        self.mask = None

    def _read_cube(self, im_path):

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

    def _crop_signal(self, lower, upper, limit_type='wavelength'):

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
        self.signal.crop_signal1D(lower_index, upper_index)

        print('Cropped signal')
        print('wavelength: {} - {}'.format(lower, upper, self.hdr['wavelength units']))
        print('index: {} - {}'.format(lower_index, upper_index))

        print(self.signal.axes_manager)

    def _set_data(self, signal):

        self.Z = signal.data
        self.y, self.x, self.ch = self.Z.shape

    def plot(self):

        self.signal.plot()
        plt.show()

    def crop_image(self, new_size, centre=None):

        if centre is None:
            y, x, ch = self.signal.data.shape
        else:
            y, x = centre

        # Calculate the indices to crop the image to from the new size
        x_mid = x / 2
        y_mid = y / 2
        x_min = x_mid - new_size / 2
        x_max = x_mid + new_size / 2
        y_min = y_mid - new_size / 2
        y_max = y_mid + new_size / 2

        # Crop the image
        self.signal.crop('x', x_min, x_max)
        self.signal.crop('y', y_min, y_max)

        self._set_data(self.signal)

        print('Cropped to {} x {}'.format(new_size, new_size))
        print(self.signal.axes_manager)

    def preprocess_spectra(self, rescale=True, normalise=False):

        print('Preprocessing spectra...')

        data = self.Z.copy()
        y = self.y
        x = self.x
        ch = self.ch

        # sum spectra to 2D array
        temp = np.sum(data, axis=2)

        # make array of indicies of nonzero elements
        spectra_index_arr = np.transpose(np.nonzero(temp)).tolist()

        # reshape spectra into n_spectra x l_spectra and remove all zero spectra (outside the image boundaries)
        spectra_arr = np.reshape(data, (x * y, ch))
        spectra_arr = spectra_arr[~np.all(spectra_arr, axis=1) == 0]

        # rescale the spectra
        if rescale:
            spectra_arr -= np.mean(spectra_arr, axis=1, keepdims=True)
            spectra_arr /= np.std(spectra_arr, axis=1, keepdims=True)

        if normalise == 'l1':
            spectra_arr = normalize(spectra_arr, norm='l1')

        if normalise == 'l2':
            spectra_arr = normalize(spectra_arr, norm='l2')

        # make mask from the summed array
        mask = np.zeros_like(temp)
        mask[temp == 0] = 1

        print('{} null spectra'.format(x * y - len(spectra_index_arr)))
        print('{} spectra'.format(len(spectra_index_arr)))
        print('spectra_arr shape: {}'.format(spectra_arr.shape))

        self.spectra_index_arr = spectra_index_arr
        self.spectra_arr = spectra_arr
        self.mask = mask


class DecomposeCube(Cube):

    def run_pca(self, plot=True):

        print('Running PCA...')
        self.signal.decomposition()
        self.signal.learning_results.summary()

        if plot:
            self.signal.plot_explained_variance_ratio(threshold=0.00001, xaxis_type='number')  # 1 thousandth
            plt.show()
            self.signal.plot_decomposition_results()
            plt.show()

    def run_ica(self, n_components, plot=True):

        print('Running ICA...')
        self.signal.blind_source_separation(n_components)
        self.signal.learning_results.summary()

        if plot:
            self.signal.plot_bss_results()
            plt.show()

    def build_signal_from_decomposition(self, n_components, plot=True):

        signal_model = self.signal.get_decomposition_model(n_components)

        self.signal = signal_model
        super()._set_data(signal_model)

        print('Built signal from {} components'.format(n_components))

        if plot:
            super().plot()

class ClusterCube(DecomposeCube):

    def rearrange_clusters_into_label_arr(self, clusters):

        # Only needed to rearrange the pyclustering label outut!
        print('Number of clusters: {}'.format(len(clusters)))

        indicies = self.spectra_index_arr
        clusters = clusters

        label_arr = np.zeros((self.spectra_arr.shape[0]))

        for label, spectra_list in enumerate(clusters):
            for spectrum_index in spectra_list:
                label_arr[spectrum_index] = label

        label_mask = np.zeros((self.y, self.x))

        for i, label in enumerate(label_arr):
            (y, x) = indicies[i]
            label_mask[y, x] = label

        label_mask = ma.array(data=label_mask, mask=self.mask)

        return label_arr, label_mask

    @timeit
    def k_means(self, n_clusters, plot=True, savepath=None):

        spectra_arr = self.spectra_arr

        # initialize centers
        initial_centers = kmeans_plusplus_initializer(spectra_arr, n_clusters).initialize()

        # create instance of K-Means with centers
        kmeans_instance = kmeans(spectra_arr, initial_centers)
        kmeans_instance.process()

        # get clusters and process into label_arr
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()
        centers = np.array(centers)

        label_arr, label_mask = self.rearrange_clusters_into_label_arr(clusters)

        title = 'K-Means, n_clusters: {}'.format(n_clusters)

        signal_data = self.Z.copy()
        signal_data = np.sum(signal_data, axis=2)
        signal_data -= np.mean(signal_data)
        signal_data /= np.std(signal_data)

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        # fig.suptitle('')

        axes[0].set_title('Mean')
        axes[0].set_ylabel('pixels / px')
        axes[0].set_xlabel('pixels / px')
        axes[0].imshow(signal_data)

        # Reset matplotlib colour cycle
        axes[1].set_prop_cycle(None)
        axes[1].set_title(title)
        axes[1].set_ylabel('pixels / px')
        axes[1].set_xlabel('pixels / px')
        axes[1].imshow(signal_data)
        axes[1].imshow(label_mask, cmap='Set1', alpha=0.4)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.95)

        if savepath is not None:
            fig.savefig(savepath)

        if plot:
            plt.show()
        else:
            plt.close()

        return label_arr, centers

    @timeit
    def plot_elbow(self, k_min=1, k_max=15, plot=True, savepath=None):

        # run elbow
        elbow_instance = elbow(self.spectra_arr, k_min, k_max)
        elbow_instance.process()

        # get the recommended amount of clusters
        n_clusters = elbow_instance.get_amount()
        print('Recommended n_clusters: {}'.format(n_clusters))

        # get wce and make the k range to plot
        wce = elbow_instance.get_wce()
        x = range(k_min, k_max)

        # make the elbow plot
        plt.figure(figsize=(5, 5))
        plt.title('Elbow plot, {} - {} clusters'.format(k_min, k_max))
        plt.ylabel('wce')
        plt.xlabel('k')
        plt.plot(x, wce, marker='x', linewidth=1)
        plt.plot(n_clusters, wce[n_clusters - 1], marker='x', c='r')
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath)

        if plot:
            plt.show()
        else:
            plt.close()

        plt.show()

        return n_clusters

if __name__ == '__main__':

    src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

    # river basin
    hdr_name = 'frt00003bfb_07_if166j_mtr3.hdr'
    im_name = 'frt00003bfb.tif'

    cube = ClusterCube(src_dir,
                hdr_name,
                im_name,
                wavelength_min=None,
                wavelength_max=2800)
    # cube.plot()

    # cube.run_pca(plot=True)
    # cube.run_ica(10, plot=True)
    # cube.build_signal_from_decomposition(5, plot=True)

    # cube.crop_image(new_size=100)
    # cube.plot()

    cube.preprocess_spectra(rescale_spectra=True)

    n_clusters = 4
    # n_clusters = cube.plot_elbow(k_min=1, k_max=15, plot=True)
    label_arr, centers = cube.k_means(n_clusters, plot=True)

    print(label_arr.shape)
    print(centers.shape)

    plt.figure()
    for i in range(centers.shape[0]):
        plt.plot(range(centers.shape[1]), centers[i])
    plt.show()

