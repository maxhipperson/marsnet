from cubes import *

src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

# river basin

names = {
    'river': ['frt00003bfb_07_if166j_mtr3.hdr', 'frt00003bfb.tif'],
    'crater': ['frt00009a16_07_if166j_mtr3.hdr', 'frt00009a16.tif'],
    'big crater': ['frt00009a9a_07_if166j_mtr3.hdr', 'frt00009a9a.tif'],
    'who knows': ['hrl000054bb_07_if182j_mtr3.hdr', 'hr1000054bb.tif']
}

# img = 'river'
# img = 'crater'
# img = 'big crater'
img = 'who knows'

cube = ClusterCube(src_dir,
                   names[img][0],
                   names[img][1],
                   wavelength_min=None,
                   wavelength_max=2800)
# cube.plot()

# cube.run_pca(plot=True)
# cube.run_ica(10, plot=True)
# cube.build_signal_from_decomposition(5, plot=True)

# cube.crop_image(new_size=100)
# cube.plot()

cube.preprocess_spectra(rescale=True,
                        # normalise='l1'
                        )

n_clusters = 4
# n_clusters = cube.plot_elbow(k_min=1, k_max=15, plot=True)
label_arr, centers = cube.k_means(n_clusters, plot=True)

print(label_arr.shape)
print(centers.shape)

plt.figure()
for i in range(centers.shape[0]):
    plt.plot(range(centers.shape[1]), centers[i])
plt.show()
