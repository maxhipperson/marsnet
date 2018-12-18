from cubes import *

src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync'

files = {
    'Mawrth Vallis': ['frt00003bfb_07_if166j_mtr3.hdr', 'frt00003bfb.tif'],
    'Oxia Planum': ['frt00009a16_07_if166j_mtr3.hdr', 'frt00009a16.tif'],
    'Jezero Crater': ['frt00005c5e_07_if166j_mtr3.hdr', 'frt00005c5e.tif'],
    'Source Crater 1': ['', ''],
    'Source Crater 2': ['', '']
}

img = 'Mawrth Vallis'
# img = 'Oxia Planum'
# img = 'Jezero Crater'
# img = 'Source Crater 1'
# img = 'Source Crater 2'

# savedir = 'raw'
savedir = 'rescale'
# savedir = 'pca_model_99.99'

# crop = 100
# crop = 400
crop = None

if crop is None:
    savedir = files[img][1].split('.')[0] + '_' + savedir
else:
    savedir = files[img][1].split('.')[0] + '_' + savedir + '_{}_crop'.format(crop)

savedir = os.path.join('/Users/maxhipperson/Documents/Year 4/marsnet/results', savedir)

# save = True
save = False

if save:
        try:
            os.makedirs(savedir)
        except OSError:
            pass


cube = ClusterCube(src_dir,
                   files[img][0],
                   files[img][1],
                   wavelength_min=None,
                   # wavelength_min=1000,
                   wavelength_max=2800,
                   # wavelength_max=None,
                   savedir=savedir
                   )
# cube.plot()

# cube.run_pca(plot=False)
# cube.run_ica(30, plot=True)
# cube.build_signal_from_decomposition([1, 2, 3, 4], plot=False)

if crop is not None:
    cube.crop_image(crop)
    # cube.plot()

cube.preprocess_spectra(
    # process='rescale',
    # process='l1,
    # process='l2'
    process='chem'
)

n_clusters = 4
# n_clusters = cube.elbow(
#     k_min=1,
#     k_max=15,
#     plot=False,
#     save=save
# )

label_arr, centers = cube.k_means(n_clusters,
                                  # plot=False,
                                  save=save
                                  )
