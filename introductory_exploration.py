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
from sklearn.cluster import SpectralClustering, KMeans
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

print('\nLoaded {}'.format(hdr_filename))
print('\twavelength units - {}'.format(header['wavelength units']))
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

# Set wavelength bounds to crop the spectrum to
crop_spectra = True
# crop_spectra = False

# im.plot()
# plt.show()

if crop_spectra:
    
    lower = 1000
    upper = 2800
    
    # find the index of the boundary wavelengths in the header
    wavelength = np.array(header['wavelength'])
    lower_index = np.argmax(wavelength >= lower)
    upper_index = np.argmax(wavelength > upper) - 1
    
    # crop the signal to the specified range
    im.crop_signal1D(lower_index, upper_index)
    
    # crop to central section of image
    im.crop('x', 120, 650)
    im.crop('y', 120, 650)

    print('\nImage cropped!')
    print('\n', im.axes_manager)

im.plot()
plt.show()


###############
# PCA and ICA #
###############


# do_PCA = True
do_PCA = False

# do_ICA = True
do_ICA = False

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

print('\nExtracting data and reshaping array')

data = im.data
old_shape = data.shape
print('Image array shape - {}'.format(old_shape))

y, x, ch = old_shape

n_clusters = 10  # TODO add user input
print('Number of clusters - {}'.format(n_clusters))

# get clusters

# flatten the data
flat = np.reshape(data, (x*y, ch))
print('Flattened array shape - {}'.format(flat.shape))

# for i in range(old_shape[0]):
#     y = i
#     for j in range(old_shape[1]):
#         x = j
#         
#         old_spectrum = data[i, j, :]
#         new_spectrum = flat[y * old_shape[0] + x, :]
#         
#         issame = np.array_equal(old_spectrum, new_spectrum)
#         if not issame:
#             print('AHHHH')

# rescale each spectrum
flat -= flat.mean(axis=1, keepdims=True)
flat /= flat.std(axis=1, keepdims=True)

# Spectral clustering
# print('Running Spectral clustering...')
# start = time()
# sc_label_arr = SpectralClustering(n_clusters=n_clusters,
#                                # assign_labels='discretize',  # TODO set to different method
#                                n_jobs=-1).fit_predict(flat)
# print('Spectral clustering done - runtime - {:f}s'.format(time() - start))
#
# sc_label_arr = sc_label_arr.reshape((y, x))
# print('Label array shape - {}'.format(sc_label_arr.shape))
sc_label_arr = np.zeros((y, x))

# K-Means clustering
print('Running K-Means Clustering...')
start = time()
km_label_arr = KMeans(n_clusters=n_clusters,
                       n_jobs=-1).fit_predict(flat)

print('K-Means done - runtime - {:f}s'.format(time() - start))

km_label_arr = km_label_arr.reshape((y, x))
print('Label array shape - {}'.format(km_label_arr.shape))

# plt.figure()
# plt.imshow(label_arr)
# plt.show()

# label_max = np.max(label_arr)
# label_cube = np.zeros((x, y, label_max + 1))
#
# for i in range(label_max + 1):
#     label_cube[:, :, i][i == label_arr] = 1

# label_mask = np.sum(label_cube, axis=2)

# plot mean image and label array
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
fig.suptitle('Plots')

ax1.set_title('Mean plot')
im1 = ax1.imshow(plot_mean_img(data))
fig.colorbar(im1, ax=ax1)

ax2.set_title('Spectral')
ax2.imshow(plot_mean_img(data), alpha=0.4)
im2 = ax2.imshow(sc_label_arr, cmap='Set1', alpha=0.4)

ax3.set_title('K means')
ax3.imshow(plot_mean_img(data), alpha=0.4)
im3 = ax3.imshow(km_label_arr, cmap='Set1', alpha=0.4)
# fig.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

