import matplotlib
matplotlib.use('Qt5Agg')
import os
import config as cfg
import hyperspy.api as hs
from skimage import io
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import process_mars_pca
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

head, im, signal_model = process_mars_pca.main(do_pca=False, do_ica=False, reconstruct_signal=False)

if signal_model not None:
    im_cluster = signal_model
    print('Using reconstructed signal model')
else:
    im_cluster = im.copy()


#########################
# Crop signal and image #
#########################


# crop image for clustering
# im_cluster = im.inav[310:330, 470:490]  # 20 x 20
im_cluster = im.inav[290:350, 450:510]  # 60 x 60
# im_cluster = im.inav[270:370, 430:530]  # 100 x 100
# im_cluster = im.inav[250:390, 410:550]  # 140 x 140
# im_cluster = im.inav[120:650, 120:650]  #

# crop image
im = im.inav[120:650, 120:650]

print('\nImage cropped for cluster identification')
print('\n', im_cluster.axes_manager)

im_cluster.plot()
plt.show()


##############
# Clustering #
##############


n_clusters = 12  # TODO add user input?
print('\nNumber of clusters - {}'.format(n_clusters))

algorithms = {
    'KMeans': KMeans,
    # 'SpectralClustering': SpectralClustering,
    # 'AgglomerativeClustering': AgglomerativeClustering,
    # 'DBSCAN': DBSCAN,
    # 'Birch': Birch
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

data = im_cluster.data
y, x, ch = data.shape
print('Image array shape - {}'.format(data.shape))

# flatten data and rescale spectra
flat = np.reshape(data, (x*y, ch))
flat =rescale_array(flat)
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
axes[0].imshow(plot_mean_img(data))
# fig.colorbar(im, ax=ax1)

for i, (key, array) in enumerate(label_arr.items()):
    axes[i + 1].set_title(key)
    axes[i + 1].imshow(plot_mean_img(data))
    axes[i + 1].imshow(array, cmap='Set1', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
