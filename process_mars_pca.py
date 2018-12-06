import os
import config as cfg
import hyperspy.api as hs
import matplotlib.pyplot as plt
from utils import *


def main(do_pca, do_ica, reconstruct_signal):

    ##############################
    # Load image and read header #
    ##############################


    image_id = 'frt00003bfb'

    hdr_filename = image_id + '_07_if166j_mtr3.hdr'
    hdr_path = os.path.join(cfg.DATA_DIR, hdr_filename)

    im_filename = image_id + '_data_cube.tif'
    im_path = os.path.join(cfg.DATA_DIR, im_filename)

    header, im = read_data_cube(hdr_path, im_path)
    crop_signal(im, header, lower=1000, upper=2800)

    im.plot()
    plt.show()

    #########################
    # Crop signal and image #
    #########################

    # crop image
    # im = im.inav[120:650, 120:650]

    # print('\n', im_cluster.axes_manager)

    # im_cluster.plot()
    # plt.show()

    ###############
    # PCA and ICA #
    ###############

    signal_model = None

    if do_pca:

        im.decomposition()
        im.learning_results.summary()

        im.plot_explained_variance_ratio(threshold=1 - 0.999)

        # plt.savefig('foo.png')
        # plt.show()

        im.plot_decomposition_results()
        plt.show()

        # im.plot_decomposition_factors()
        # im.plot_decomposition_loadings()
        # plt.show()

        # combine the pca componants into a signal to perform clustering
        if reconstruct_signal and not do_ica:

            while True:
                try:
                    n_components = int(input('Set n_components for decomposition model: '))
                except ValueError:
                    print('Not an integer!')
                    continue
                else:
                    print('n_components - {}'.format(n_components))
                    break

            # n_componants = [0, 1, 2, 4, 6]
            pca_model = im.get_decomposition_model(n_components)

            pca_model.plot()
            plt.show()

        if do_ica:

            while True:
                try:
                    n_components = int(input('Set n_components for ICA decomposition: '))
                except ValueError:
                    print('Not an integer!')
                    continue
                else:
                    print('n_components - {}'.format(n_components))
                    break

            im.blind_source_separation(number_of_components=n_components)
            im.learning_results.summary()

            im.plot_bss_results()
            plt.show()

            # im.plot_bss_factors()
            # im.plot_bss_loadings()
            # plt.show()

            if reconstruct_signal:

                while True:
                    try:
                        n_components = int(input('\nSet n_components for decomposition model:\n'))
                    except ValueError:
                        print('Not an integer!')
                        continue
                    else:
                        print('n_components - {}'.format(n_components))
                        break

                # n_componants = [0, 1, 2, 4, 6]
                signal_model = im.get_decomposition_model(n_components)

                signal_model.plot()
                plt.show()

    return header, im, signal_model

if __name__ == '__main__':

    do_pca = True
    do_ica = False
    reconstruct_signal = False

    header, im, signal_model = main(do_pca=do_pca,
                                    do_ica=do_ica,
                                    reconstruct_signal=reconstruct_signal)
