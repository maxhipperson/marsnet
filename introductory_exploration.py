import matplotlib
matplotlib.use('Qt5Agg')
import os
import config as cfg
import hyperspy.api as hs
from skimage import io
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import envi_header
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch
from time import time
# import seaborn as sns
# sns.set()
# from matplotlib.colors import ListedColormap


def rescale_array(array):

    array -= array.mean(axis=1, keepdims=True)
    array /= array.std(axis=1, keepdims=True)

    return array

def plot_mean_img(cube):

    cube = np.sum(cube, axis=2)
    cube -= np.mean(cube)
    cube /= np.std(cube)

    return cube


##############################
# Load image and read header #
##############################


image_id = 'frt00003bfb'

# read header file
hdr_filename = image_id + '_07_if166j_mtr3.hdr'
hdr_filepath = os.path.join(cfg.DATA_DIR, hdr_filename)
header = envi_header.read_hdr_file(hdr_filepath)

print('\nLoaded {}:'.format(hdr_filename))
for key in sorted(header.keys()):
    print('\t{}'.format(key))
print('\n\twavelength units - {}'.format(header['wavelength units']))
print('\tdata ignore value - {}'.format(header['data ignore value']))

# read tiff file
img_filename = image_id + '_data_cube.tif'
img_filepath = os.path.join(cfg.DATA_DIR, img_filename)
img = io.imread(img_filepath)

print('\nLoaded {}'.format(img_filename))

c, w, h = img.shape
img = img.transpose(1, 2, 0)

# remove ignored values
img[img == header['data ignore value']] = 0

'''
Import the image into a signal class object
'''

(y, x, ch) = img.shape

# plt.imshow(img[:, :, :3])
# plt.show()

axes_x = {'name': 'x', 'size': x, 'units': 'px'}
axes_y = {'name': 'y', 'size': y, 'units': 'px'}
axes_ch = {'name': 'wavelength band', 'size': ch, 'units': 'index'}

# convert image to signal object
im = hs.signals.Signal1D(img, axes=[axes_y, axes_x, axes_ch])
print('\n', im.axes_manager)

# im.plot()
# plt.show()


#########################
# Crop signal and image #
#########################


# wavelength range to crop to in nm
lower = 1000
upper = 2800

# find the index of the boundary wavelengths in the header
wavelength = np.array(header['wavelength'])
lower_index = np.argmax(wavelength >= lower)
upper_index = np.argmax(wavelength > upper) - 1

# crop the signal to the index range from header
im.crop_signal1D(lower_index, upper_index)

print('\nImage cropped for cluster identification')

# crop image for clustering
# im_cluster = im.inav[310:330, 470:490]  # 20 x 20
im_cluster = im.inav[290:350, 450:510]  # 60 x 60
# im_cluster = im.inav[270:370, 430:530]  # 100 x 100
# im_cluster = im.inav[250:390, 410:550]  # 140 x 140
# im_cluster = im.inav[120:650, 120:650]  #

# crop image
# im = im.inav[120:650, 120:650]

print('\n', im_cluster.axes_manager)

im.plot()
plt.show()

im_cluster.plot()
plt.show()


###############
# PCA and ICA #
###############
# todo pca then reconstruct signal from pca componants and run clustering


do_PCA = True
# do_PCA = False

do_ICA = True
# do_ICA = False

# reconstruct_signal = True
reconstruct_signal = False

if do_PCA:

    im.decomposition()
    im.learning_results.summary()

    im.plot_explained_variance_ratio(threshold=1 - 0.999)
    plt.show()

    im.plot_decomposition_results()
    plt.show()

    # im.plot_decomposition_factors()
    # im.plot_decomposition_loadings()
    # plt.show()

    # combine the pca componants into a signal to perform clustering
    if reconstruct_signal:

        n_componants = 10
        pca_componants = im.get_decomposition_loadings()

        # sel;ect signal componants

        # build signal

        #


    if do_ICA:

        n_componants = 20  # TODO add user input
        im.blind_source_separation(number_of_components=n_componants)
        im.learning_results.summary()

        im.plot_bss_results()
        plt.show()

        # im.plot_bss_factors()
        # im.plot_bss_loadings()
        # plt.show()


#######################
# Spectral Clustering #
#######################

n_clusters = 12  # TODO add user input?
print('\nNumber of clusters - {}'.format(n_clusters))

algorithms = {
    'KMeans': KMeans,
    'SpectralClustering': SpectralClustering,
    # 'AgglomerativeClustering': AgglomerativeClustering,
    # 'DBSCAN': DBSCAN,
    'Birch': Birch
}

print('\nExtracting data and reshaping array')

cluster_train = im_cluster.data
y, x, ch = cluster_train.shape
print('Image array shape - {}'.format(cluster_train.shape))

# flatten data and rescale spectra
flat = np.reshape(cluster_train, (x*y, ch))
flat -= flat.mean(axis=1, keepdims=True)
flat /= flat.std(axis=1, keepdims=True)
# print('Flattened array shape - {}'.format(flat.shape))

label_arr = {}
for key in algorithms.keys():

    print('Running {}...'.format(key))
    start = time()
    algorithms[key] = algorithms[key](n_clusters=n_clusters)
    label_arr[key] = algorithms[key].fit_predict(X=flat)
    print('{} done - runtime - {:f}s'.format(key, time() - start))
    label_arr[key] = label_arr[key].reshape((y, x))
    print(label_arr[key].shape)

# plot mean image and label array from clustering
fig, axes = plt.subplots(1, len(label_arr) + 1, figsize=(15, 10))
fig.suptitle('Plots')

axes[0].set_title('Mean')
axes[0].imshow(plot_mean_img(cluster_train))
# fig.colorbar(im, ax=ax1)

for i, (key, array) in enumerate(label_arr.items()):

    axes[i + 1].set_title(key)
    axes[i + 1].imshow(plot_mean_img(cluster_train))
    axes[i + 1].imshow(array, cmap='Set1', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


#####################################################
# predict clustering and plot mean image and labels #
#####################################################


print('\nExtracting data and reshaping array')

cluster_test = im.data
y, x, ch = cluster_test.shape
print('Image array shape - {}'.format(cluster_test.shape))

# flatten data and rescale spectra
flat = np.reshape(cluster_test, (x*y, ch))
flat -= flat.mean(axis=1, keepdims=True)
flat /= flat.std(axis=1, keepdims=True)
# print('Flattened array shape - {}'.format(flat.shape))

label_arr = {}

for key in algorithms.keys():
    print('Running {}...'.format(key))
    start = time()
    label_arr[key] = algorithms[key].predict(X=flat)
    print('{} done - runtime - {:f}s'.format(key, time() - start))
    label_arr[key] = label_arr[key].reshape((y, x))
    print(label_arr[key].shape)

# plot mean image and label array from clustering
fig, axes = plt.subplots(1, len(label_arr) + 1, figsize=(15, 10))
fig.suptitle('Plots')

axes[0].set_title('Mean')
axes[0].imshow(plot_mean_img(cluster_test))
# fig.colorbar(im, ax=ax1)

for i, (key, array) in enumerate(label_arr.items()):
    axes[i + 1].set_title(key)
    axes[i + 1].imshow(plot_mean_img(cluster_test))
    axes[i + 1].imshow(array, cmap='Set1', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
