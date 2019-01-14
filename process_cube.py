from cubes import *

src_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync/hs_imgs'
dst_dir = '/Users/maxhipperson/Documents/Year 4/marsnet/data.nosync/hs_results'

files = {
    'mawrth_vallis': ['frt00003bfb_07_if166j_mtr3.hdr', 'frt00003bfb.tif'],
    'oxia_planum': ['frt00009a16_07_if166j_mtr3.hdr', 'frt00009a16.tif'],
    'jezero_crater': ['frt00005c5e_07_if166j_mtr3.hdr', 'frt00005c5e.tif'],
    'source_crater_1': ['frt00009568_07_if163j_mtr3.hdr', 'frt00009568.tif'],
    'source_crater_2': ['frt000083dc_07_if164j_mtr3.hdr', 'frt000083dc.tif']
}

##############################
# Choose hs image

img = 'mawrth_vallis'
# img = 'oxia_planum'
# img = 'jezero_crater'
# img = 'source_crater_1'
# img = 'source_crater_2'

# save = True
save = False

##############################

# plot = False
plot = True

# plot_decomposition = False
plot_decomposition = True

# plot_clustering = False
plot_clustering = True

##############################

wavelength_min = None
# wavelength_max = None

# wavelength_min = 730
# wavelength_min = 1000
# wavelength_min = 2820
# wavelength_min = 3500

wavelength_max = 2800

##############################

# Can't do as well as cropping the signal... todo sort?

# crop_section = True
crop_section = False

crop_section_min = 2800
# crop_section_max = 3000
crop_section_max = 3500

##############################

# pca = False
pca = True

# ica = False
ica = True

n_components_ica = 20

build_signal = False
# build_signal = True

n_components_model = 5

##############################

crop = None
# crop = 100
# crop = 200
# crop = 300
# crop = 400

##############################

# preprocess = 'rescale'
# preprocess='l1'
# preprocess='l2'
preprocess = 'chem'

##############################

k_min = 1
k_max = 15

# n_clusters = 'elbow'
n_clusters = 4
# n_clusters = 5
# n_clusters = 6

##############################
# Run script

savedir = [dst_dir, img]

cond1 = wavelength_min is None
cond2 = wavelength_max is None

if crop_section:
    savedir.append('crop_sec_{}-{}'.format(crop_section_min, crop_section_max))
else:
    savedir.append('wl_{}-{}'.format(wavelength_min, wavelength_max))

# todo Add dirname for removing wl section

if build_signal:
    if pca:
        savedir.append('pca_model_{}_components'.format(n_components_model))
    elif ica:
        savedir.append('ica_model_{}_components'.format(n_components_model))

if crop is not None:
    savedir.append('crop_{}'.format(crop))
else:
    savedir.append('no_crop'.format(crop))

savedir.append('preprocess_{}'.format(preprocess))

savedir = os.path.join(*savedir)

if save:
        try:
            os.makedirs(savedir)
            print('Made {}'.format(savedir))
        except OSError:
            print(OSError('{} exists!'.format(savedir)))

# Crop the signal range in the class instantiation or
# remove a section from the signal with the method

cube = ClusterCube(src_dir,
                   files[img][0],
                   files[img][1],
                   wavelength_min=wavelength_min,
                   wavelength_max=wavelength_max,
                   savedir=savedir
                   )
if plot:
    cube.plot()

if crop_section:
    cube.remove_section_from_signal(crop_section_min, crop_section_max)
    if plot:
        cube.plot()

if pca:
    cube.run_pca(plot=plot_decomposition)
    if ica:
        cube.run_ica(n_components_ica, plot=plot_decomposition)
    if build_signal:
        cube.build_signal_from_decomposition(n_components_model, plot=plot_decomposition)

if crop is not None:
    cube.crop_image(crop)
    if plot:
        cube.plot()

cube.preprocess_spectra(process=preprocess)


# If set to use elbow then run elbow to get the estimated number of clusters
if n_clusters is 'elbow':
    n_clusters = cube.elbow(k_min=1, k_max=15, plot=plot_clustering, save=save)

label_arr, centers = cube.k_means(n_clusters, plot=plot_clustering, save=save)
