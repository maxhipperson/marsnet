from cubes import *

src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

files = {
    'river': ['frt00003bfb_07_if166j_mtr3.hdr', 'frt00003bfb.tif'],
    'crater': ['frt00009a16_07_if166j_mtr3.hdr', 'frt00009a16.tif'],
    'big crater': ['frt00009a9a_07_if166j_mtr3.hdr', 'frt00009a9a.tif'],
    'who knows': ['hrl000054bb_07_if182j_mtr3.hdr', 'hr1000054bb.tif']
}

# img = 'river'
# img = 'crater'
img = 'big crater'
# img = 'who knows'

savedir = '/Users/maxhipperson/Documents/Year 4/marsnet/results'

cube = ClusterCube(src_dir,
                   files[img][0],
                   files[img][1],
                   wavelength_min=None,
                   wavelength_max=2800,
                   savedir=savedir
                   )
# cube.plot()

# cube.run_pca(plot=False)
# cube.run_ica(10, plot=True)
# cube.build_signal_from_decomposition(1, plot=False)

cube.crop_image(new_size=368)
# cube.plot()

cube.preprocess_spectra(rescale=True,
                        # normalise='l2'
                        )

n_clusters = 5
# n_clusters = cube.elbow(k_min=1, k_max=15, plot=True)
label_arr, centers = cube.k_means(n_clusters,
                                  # plot=True,
                                  # save=True
                                  )
