
import os
import config as cfg
from time import time
import hyperspy.api as hs
import matplotlib.pyplot as plt
from utils import *
import process_mars_pca

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def rescale_array(array):

    array -= array.mean(axis=1, keepdims=True)
    array /= array.std(axis=1, keepdims=True)

    return array

def plot_mean_img(cube):

    cube = np.sum(cube, axis=2)
    cube -= np.mean(cube)
    cube /= np.std(cube)

    return cube

head, im, signal_model = process_mars_pca.main(do_pca=True, do_ica=False, reconstruct_signal=False)

if not signal_model is None:
    im_cluster = signal_model
    print('Using reconstructed signal model')
else:
    im_cluster = im.copy()


#########################
# Crop signal and image #
#########################


# crop image for clustering
# im_cluster = im_cluster.inav[310:330, 470:490]  # 20 x 20
# im_cluster = im_cluster.inav[290:350, 450:510]  # 60 x 60
# im_cluster = im_cluster.inav[270:370, 430:530]  # 100 x 100
im_cluster = im_cluster.inav[250:390, 410:550]  # 140 x 140
# im_cluster = im_cluster.inav[120:650, 120:650]  #

# crop image
im = im.inav[120:650, 120:650]

print('\nImage cropped for cluster identification')
print('\n', im_cluster.axes_manager)

im_cluster.plot()
plt.show()

print('\nExtracting data and reshaping array')




##################
# Run clustering #
##################


# n_clusters = 12  # TODO add user input?
# print('\nNumber of clusters - {}'.format(n_clusters))

# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
n_initial_clusters = 8
initial_centers = kmeans_plusplus_initializer(flat, n_initial_clusters).initialize()

# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
# number of clusters that can be allocated is 20.
xmeans_instance = xmeans(flat, initial_centers, 50, ccore=False)
xmeans_instance.process()

clusters = xmeans_instance.get_clusters()

print('Number of clusters - {}'.format(len(clusters)))

label_arr = np.zeros((flat.shape[0]))
print('Flattened label array shape - {}'.format(label_arr.shape))

for cluster_id, list in enumerate(clusters):
    for index in list:
        label_arr[index] = cluster_id

print(label_arr.shape)

label_arr = label_arr.reshape((y, x))
print(label_arr.shape)

# plot mean image and label array from clustering
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Plots')

axes[0].set_title('Mean')
axes[0].imshow(plot_mean_img(cluster_train))
# fig.colorbar(im, ax=ax1)

axes[1].set_title('X-means')
axes[1].imshow(plot_mean_img(cluster_train))
axes[1].imshow(label_arr, cmap='Set1', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('clustering_x_means_initial_{}_n_clusters_{}.png'.format(initial_centers, len(clusters)))
plt.show()


# #####################################################
# # predict clustering and plot mean image and labels #
# #####################################################
#
#
# print('\nExtracting data and reshaping array')
#
# data = im_cluster.data
# y, x, ch = data.shape
# print('Image array shape - {}'.format(data.shape))
#
# # flatten data and rescale spectra
# flat = np.reshape(data, (x*y, ch))
# flat =rescale_array(flat)
# # print('Flattened array shape - {}'.format(flat.shape))
#
# label_arr = {}
#
# for key in algorithms.keys():
#     print('Running {}...'.format(key))
#     start = time()
#     label_arr[key] = algorithms[key].predict(X=flat)
#     print('{} done - runtime - {:f}s'.format(key, time() - start))
#     label_arr[key] = label_arr[key].reshape((y, x))
#     print(label_arr[key].shape)
#
# # plot mean image and label array from clustering
# fig, axes = plt.subplots(1, len(label_arr) + 1, figsize=(15, 10))
# fig.suptitle('Plots')
#
# axes[0].set_title('Mean')
# axes[0].imshow(plot_mean_img(data))
# # fig.colorbar(im, ax=ax1)
#
# for i, (key, array) in enumerate(label_arr.items()):
#     axes[i + 1].set_title(key)
#     axes[i + 1].imshow(plot_mean_img(data))
#     axes[i + 1].imshow(array, cmap='Set1', alpha=0.4)
#
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()
