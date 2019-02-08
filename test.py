import os
import json
import numpy as np
import random
import config as cfg

from torch.utils.data import TensorDataset, DataLoader

from nets import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def load_data(data_dir, data_files):

    print()
    print('Loaded data files')

    data = {}

    for data_file in data_files:

        array = np.load(os.path.join(data_dir, data_file))
        data[data_file[:-4]] = array

        print()
        print('Loaded {}, array of shape: {}'.format(data_file, array.shape))

    return data


def load_params(param_dir, param_file):

    print()
    print('=' * 50)
    print('Loaded {}'.format(param_file))

    with open(os.path.join(param_dir, param_file), 'r') as file:
        params = json.load(file)

    for param, value in params.items():

        # print()
        print('{}: {}'.format(param, value))

    return params


def get_abundances():
    raise NotImplementedError


def get_endmembers(model):

    weights = model.decoder[0].weight
    # weights = model.decoder[-1].weight
    weights = weights.detach().numpy()

    return weights


def predict(model, x):

    with torch.no_grad():
        model.eval()

        x = torch.Tensor(x)

        encoded, out = model(x)

    return encoded.numpy(), out.numpy()


def plot_mean_image(data):

    fig, ax = plt.subplots()
    # fig.suptitle('')

    mean_image = np.mean(data, axis=2)

    ax.set_title('Mean Image', fontsize='medium')
    ax.set_ylabel('pixels / px')
    ax.set_xlabel('pixels / px')
    ax.imshow(mean_image, cmap='gray')

    plt.tight_layout()
    plt.show()


def plot_endmembers(wavelengths, endmembers):

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title('Extracted Endmembers', fontsize='medium')

    for i in range(endmembers.shape[1]):
        ax.plot(wavelengths / 1000, endmembers[:, i], linewidth=1, label='endmember {}'.format(i))
    # for i in range(endmembers.shape[0]):
    #     ax.plot(wavelengths / 1000, endmembers[i], linewidth=1, label='endmember {}'.format(i))

    ax.set_ylabel('')
    ax.set_xlabel(r'Wavelength / $\mu$m')
    fig.legend(fontsize='small')

    plt.tight_layout()
    plt.show()


def plot_input_and_output(wavelengths, input, output):

    input = np.squeeze(input)
    output = np.squeeze(output)

    fig, ax = plt.subplots()

    ax.set_title('', fontsize='medium')
    ax.plot(wavelengths / 1000, input, linewidth=1, label='input'.format(0))
    ax.plot(wavelengths / 1000, output, linewidth=1, label='output'.format(0))
    ax.set_ylabel('')
    ax.set_xlabel(r'Wavelength / $\mu$m')
    fig.legend(fontsize='small')

    plt.tight_layout()
    plt.show()


def plot_abundance_map(abundances):
    raise NotImplementedError


if __name__ == '__main__':

    data_dir = cfg.DATA_DIR
    param_dir = cfg.PARAM_DIR

    param_file = 'params.json'
    # param_file = 'params_0.json'

    # model_file = 'model_epoch_0.pt'
    # model_file = 'model_epoch_5.pt'
    # model_file = 'model_epoch_10.pt'
    model_file = 'model_epoch_15.pt'
    # model_file = 'model_epoch_20.pt'

    model_path = os.path.join(param_dir, param_file[:-5], model_file)

    data_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]

    data = load_data(data_dir, data_files)
    params = load_params(param_dir, param_file)

    # Plot mean image
    plot_mean_image(data['signal'])

    model = cfg.MODEL

    # Instantiate model

    input_shape = data['wavelengths'].shape[0]

    model = model(input_shape=input_shape, **params['model'])
    model.load_state_dict(torch.load(model_path))

    for name, param in model.decoder.named_parameters():
        print(name, '\t', param.shape)

    endmembers = get_endmembers(model)

    plot_endmembers(data['wavelengths'], endmembers)

    #############
    # Run model #
    #############

    print()
    print('Running model')

    with torch.no_grad():

        model.eval()

        num_samples = 3

        for i in range(num_samples):

            x = data['spectra_arr'][np.random.randint(data['spectra_arr'].shape[0]), :]
            x = np.expand_dims(x, 0)

            abundances, output = predict(model, x)

            print()
            print(abundances)
            plot_input_and_output(data['wavelengths'], x, output)

        spectra_arr = data['spectra_arr']
        spectra_arr = torch.Tensor(spectra_arr)

        abundances, __ = model(spectra_arr)
        abundances = abundances.numpy()

        n_endmembers = abundances.shape[1]

        for i in range(n_endmembers):

            nth_abundances = abundances[:, i]

            abundance_map = np.zeros((data['signal'].shape[0], data['signal'].shape[1]))

            for j, value in enumerate(nth_abundances):
                (y, x) = data['spectra_index_arr'][j]
                abundance_map[y, x] = value

            mean_image = np.mean(data['signal'], axis=2)
            mask = np.zeros_like(mean_image)
            mask[mean_image == 0] = 1

            abundance_mask = np.ma.array(data=abundance_map, mask=mask)

            fig, ax = plt.subplots()
            # fig.suptitle('')

            ax.set_title('', fontsize='medium')
            ax.set_ylabel('pixels / px')
            ax.set_xlabel('pixels / px')

            mean_image = np.mean(data['signal'], axis=2)

            # ax.imshow(abundance_mask, alpha=0.4, cmap='cubehelix')
            ax.set_title('Abundance Map: endmember {}'.format(i), fontsize='medium')
            ax.set_ylabel('pixels / px')
            ax.set_xlabel('pixels / px')
            # ax.imshow(mean_image, cmap='gray')
            # im = ax.imshow(abundance_mask, cmap='cubehelix', alpha=0.5)
            im = ax.imshow(abundance_mask, cmap='cubehelix')
            plt.colorbar(im)

            plt.tight_layout()

        plt.show()


