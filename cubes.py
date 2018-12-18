import matplotlib
matplotlib.use('Qt5Agg')
from utils import *
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn  as sns
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric;
import hdbscan
import fitsne
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
# sns.set()
import scipy.signal
import pickle as pkl


class Cube(object):
    def __init__(self, src_dir, hdr_name, im_name, wavelength_min=None, wavelength_max=None, savedir='.'):

        hdr_path = os.path.join(src_dir, hdr_name)
        im_path = os.path.join(src_dir, im_name)

        self.hdr = read_hdr_file(hdr_path)
        self.im = self._read_cube(im_path)

        print('Loaded {}:'.format(hdr_path))
        print('Loaded {}'.format(im_path))

        self.wavelengths = np.array(self.hdr['wavelength'])
        self.signal = self.im.copy()

        self._crop_signal(lower=wavelength_min, upper=wavelength_max, limit_type='wavelength')
        self._set_data(self.signal)

        self.removed_wavelengths = None

        self.spectra_index_arr = None
        self.spectra_arr = None
        self.mask = None

        self.savedir = savedir

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
        lower_index, upper_index = self._get_index_from_wavelengths(lower, upper, limit_type)

        # crop the signal to the index range from header
        self.signal.crop_signal1D(lower_index, upper_index)
        self.wavelengths = self.wavelengths[lower_index:upper_index]

        print('Cropped signal')
        print('wavelength: {} - {} {}'.format(lower, upper, self.hdr['wavelength units']))
        print('index: {} - {}'.format(lower_index, upper_index))

        print(self.signal.axes_manager)

    def _get_index_from_wavelengths(self, lower, upper, limit_type='wavelength'):

        if limit_type == 'wavelength':

            if lower is None:
                lower_index = None
            else:
                lower_index = np.argmax(self.wavelengths >= lower)

            if upper is None:
                upper_index = None
            else:
                upper_index = np.argmax(self.wavelengths > upper) - 1

        else:
            lower_index = lower
            upper_index = upper

        return lower_index, upper_index


    def _set_data(self, signal):

        self.Z = signal.data
        self.y, self.x, self.ch = self.Z.shape

    def plot(self):

        self.signal.plot()
        plt.show()

    def remove_section_from_signal(self, lower, upper, limit_type='wavelength'):

        # find the index of the boundary wavelengths in the header
        lower_index, upper_index = self._get_index_from_wavelengths(lower, upper, limit_type)

        # crop the signal to the index range from header
        temp1 = self.signal.isig[:lower_index]
        temp2 = self.signal.isig[upper_index:]

        self.signal = hs.stack([temp1, temp2], axis=2)
        self._set_data(self.signal)

        self.wavelengths = np.delete(self.wavelengths, np.s_[lower_index:upper_index])
        self.removed_wavelengths = {'lower':lower,
                                    'upper':upper,
                                    'lower_index':lower_index,
                                    'upper_index':upper_index}

        print('Removed section from signal')
        print('removed wavelengths: {} - {} {}'.format(lower, upper, self.hdr['wavelength units']))
        print('removed indexs: {} - {}'.format(lower_index, upper_index))
        print(self.signal.axes_manager)

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

    def preprocess_spectra(self, process=None):
        """"""

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
        if process == 'rescale':
            spectra_arr -= np.mean(spectra_arr, axis=1, keepdims=True)
            spectra_arr /= np.std(spectra_arr, axis=1, keepdims=True)

        if process == 'l1':
            spectra_arr = normalize(spectra_arr, norm='l1')

        if process == 'l2':
            spectra_arr = normalize(spectra_arr, norm='l2')

        if process == 'chem':



            # apply Savitzky–Golay filter to mean spectrum and subtract from spectra
            # ffit = scipy.signal.savgol_filter(np.mean(spectra_arr, axis=0), 29, 3)
            # spectra_arr -= ffit

            # standard normal variate
            spectra_arr -= np.mean(spectra_arr, axis=1, keepdims=True)
            # spectra_arr /= np.std(spectra_arr, axis=1, keepdims=True)

            # autoscale
            spectra_arr /= np.std(spectra_arr, axis=0, keepdims=True)

        # make mask from the summed array
        mask = np.zeros_like(temp)
        mask[temp == 0] = 1

        print('{} null spectra'.format(x * y - len(spectra_index_arr)))
        print('{} spectra'.format(len(spectra_index_arr)))
        print('spectra_arr shape: {}'.format(spectra_arr.shape))

        self.spectra_index_arr = spectra_index_arr
        self.spectra_arr = spectra_arr
        self.mask = mask

    def fit_mean_spectrum(self, plot=True, save=False, data=None):

        # fit polynomial to the mean spectrum and subtract from spectra
        if data is None:
            mean = np.mean(self.spectra_arr, axis=0)
        else:
            mean = np.mean(data, axis=0)
        # coefs = np.polyfit(self.wavelengths, mean, deg=deg)
        # ffit = np.polyval(coefs, self.wavelengths)

        window = 29
        order = 3
        ffit = scipy.signal.savgol_filter(mean, window, order)

        fig, ax = plt.subplots()
        ax.plot(self.wavelengths, mean, linewidth=1, label='mean spectrum')
        # ax.plot(self.wavelengths, ffit, linewidth=1, label='polynomial of degree {}'.format(deg))
        # ax.plot(self.wavelengths, mean - ffit, linewidth=1, label='fit subtracted mean')
        ax.plot(self.wavelengths, ffit, linewidth=1, label='smoothed mean')
        ax.plot(self.wavelengths, mean - ffit, linewidth=1, label='residuals')
        # ax.set_title('Polynomial of degree {} fit to mean spectrum'.format(deg), fontsize='medium')
        ax.set_title('Savgol Filter Applied to Mean Spectrum\nOrder {}, Window {}'.format(order, window), fontsize='medium')
        ax.set_xlabel(r'Wavelength / $\mu$m')
        fig.legend(fontsize='small')

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.savedir, 'savgol_filter.png'), dpi=300)

        if plot:
            fig.show()
        else:
            fig.close()

        return ffit

    def save_spectral_data(self, savename='spectral_data.pkl'):

        data = {'wavelengths': self.wavelengths,
                'signal_arr': self.Z,
                'spectra_index_arr': self.spectra_index_arr,
                'spectra_arr': self.spectra_arr,
                'mask': self.mask}

        savepath = os.path.join(self.savedir, savename)
        if not os.path.exists(savepath):
            with open(savepath, 'wb') as file:
                pkl.dump(data, file)
            for key in data.keys():
                print('Saved {} to {}'.format(key, savepath))
        else:
            raise FileExistsError('{} exists!'.format_map(savepath))



class DecomposeCube(Cube):

    def run_pca(self, plot=True):

        print('Running PCA...')
        self.signal.decomposition()
        self.signal.learning_results.summary()

        if plot:
            # threshold = 0.001  # 99.9 % -
            # threshold = 0.0001  # 99.99 % -
            threshold = 0.00001  # 99.999 % - thousandth of a percent

            self.signal.plot_explained_variance_ratio(threshold=threshold, xaxis_type='number')  # 1 thousandth
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

        label_arr = np.zeros((self.spectra_arr.shape[0]))

        for label, spectra_list in enumerate(clusters):
            for spectrum_index in spectra_list:
                label_arr[spectrum_index] = label

        label_mask = np.zeros((self.y, self.x))

        for i, label in enumerate(label_arr):
            (y, x) = self.spectra_index_arr[i]
            label_mask[y, x] = label

        label_mask = ma.array(data=label_mask, mask=self.mask)

        return label_arr, label_mask

    @timeit
    def k_means(self, n_clusters, plot=True, save=False):

        ##############################
        # Run clustering

        # Define metric to use for k means
        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

        # define custom metric
        # def cosine_distance(a, b):
        #     return 1 - np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # metric = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)

        # create instance of K-Means with centers
        initial_centers = kmeans_plusplus_initializer(self.spectra_arr, n_clusters).initialize()
        kmeans_instance = kmeans(self.spectra_arr, initial_centers, metric=metric)
        kmeans_instance.process()

        # get clusters and process into label_arr
        clusters = kmeans_instance.get_clusters()
        centers = np.array(kmeans_instance.get_centers())

        label_arr, label_mask = self.rearrange_clusters_into_label_arr(clusters)

        signal_data = self.Z.copy()
        signal_data = np.sum(signal_data, axis=2)
        signal_data -= np.mean(signal_data)
        signal_data /= np.std(signal_data)

        ##############################
        # Plot clustering map

        palette = sns.color_palette('Set1', n_colors=n_clusters)
        cmap = ListedColormap(palette.as_hex())

        fig1, axes = plt.subplots(1, 2, figsize=(10, 6))
        # fig.suptitle('')

        axes[0].set_title('Mean Image', fontsize='medium')
        axes[0].set_ylabel('pixels / px')
        axes[0].set_xlabel('pixels / px')
        axes[0].imshow(signal_data)

        # Reset matplotlib colour cycle
        # axes[1].set_prop_cycle(None)
        axes[1].set_title('K-Means with {} clusters'.format(n_clusters), fontsize='medium')
        axes[1].set_ylabel('pixels / px')
        axes[1].set_xlabel('pixels / px')
        axes[1].imshow(signal_data)
        im = axes[1].imshow(label_mask, cmap=cmap, alpha=0.4)
        # plt.colorbar(im)

        plt.tight_layout()

        ##############################
        # Plot spectra from clustering

        fig2, axes = plt.subplots(2, 1, figsize=(6, 6))

        axes[0].set_title('Cluster Centers', fontsize='medium')
        for i in range(centers.shape[0]):
            axes[0].plot(self.wavelengths / 1000, centers[i], linewidth=1, c=palette[i], label='Cluster {}'.format(i))
        axes[0].set_ylabel('')
        axes[0].set_xlabel('')

        # fit polynomial to the mean spectrum and subtract from centers
        ffit = self.fit_mean_spectrum(plot, save)

        axes[1].set_title('Mean Subtracted Cluster Centers', fontsize='medium')

        for i in range(centers.shape[0]):

            mean_sub = centers[i] - ffit
            # mean_sub = scipy.signal.savgol_filter(mean_sub, 11, 3)  # TODO smooth mean subbed cluster centers

            axes[1].plot(self.wavelengths / 1000, mean_sub, linewidth=1, c=palette[i])

        axes[1].set_ylabel('')
        axes[1].set_xlabel(r'Wavelength / $\mu$m')

        fig2.legend(fontsize='small')
        plt.tight_layout()

        if save:
            fig1.savefig(os.path.join(self.savedir, '{}_clusters.map.png'.format(n_clusters)), dpi=300)
            fig2.savefig(os.path.join(self.savedir, '{}_clusters.spectra.png'.format(n_clusters)), dpi=300)

        if plot:
            plt.show()
        else:
            plt.close()

        return label_arr, centers

    @timeit
    def elbow(self, k_min=1, k_max=15, plot=True, save=False):

        # run elbow
        elbow_instance = elbow(self.spectra_arr, k_min, k_max)
        elbow_instance.process()

        # get the recommended amount of clusters
        n_clusters = elbow_instance.get_amount()
        print('Recommended n_clusters: {}'.format(n_clusters))

        ##############################
        # Plot elbow

        # get wce and make the k range to plot
        wce = elbow_instance.get_wce()
        x = range(k_min, k_max)

        # make the elbow plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title('Elbow plot, {} - {} clusters'.format(k_min, k_max), fontsize='medium')
        ax.set_ylabel('total within-cluster error (wce)')
        ax.set_xlabel('n_clusters (k)')
        ax.set_xticks(x)
        ax.plot(x, wce, marker='x', linewidth=1)
        ax.plot(n_clusters, wce[n_clusters - 1], marker='x', c='r')

        ax.annotate('optimal n_clusters: {}'.format(n_clusters),
                    xy=(n_clusters, wce[n_clusters - 1]),
                    textcoords='axes fraction',
                    xytext=(0.3, 0.5),
                    arrowprops=dict(arrowstyle='->'))

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.savedir, 'elbow.png'), dpi=300)

        if plot:
            plt.show()
        else:
            plt.close()

        plt.show()

        return n_clusters

if __name__ == '__main__':

    src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

    files = {
        'mawrth_vallis': ['frt00003bfb_07_if166j_mtr3.hdr', 'frt00003bfb.tif'],
        'oxia_planum': ['frt00009a16_07_if166j_mtr3.hdr', 'frt00009a16.tif'],
        'jezero_crater': ['frt00005c5e_07_if166j_mtr3.hdr', 'frt00005c5e.tif'],
        'source_crater_1': ['', ''],
        'source_crater_2': ['', '']
    }

    img = 'mawrth_vallis'
    # img = 'oxia_planum'
    # img = 'jezero_crater'
    # img = 'source_crater_1'
    # img = 'source_crater_2'

    # crop = None
    crop = 100
    # crop = 200
    # crop = 300
    # crop = 400

    if crop is None:
        savedir = img
    else:
        savedir = img + '/{}_crop'.format(crop)

    savedir = os.path.join('/Users/maxhipperson/Documents/Year 4/marsnet/results', savedir)

    try:
        os.makedirs(savedir)
    except OSError:
        pass

    cube = ClusterCube(src_dir,
                       files[img][0],
                       files[img][1],
                       # wavelength_min=730,
                       # wavelength_min=1000,
                       # wavelength_min=2820,
                       # wavelength_max=2800,
                       savedir=savedir
                       )
    cube.plot()

    if crop is not None:
        cube.crop_image(crop)
        cube.plot()

    cube.preprocess_spectra(
        # process='rescale',
        # process='l1,
        # process='l2'
        process='chem'
    )

    # cube.run_pca(plot=True)
    # cube.run_ica(10, plot=True)
    # cube.build_signal_from_decomposition(5, plot=True)

    # cube.crop_image(new_size=100)
    # cube.plot()

    cube.save_spectral_data()

