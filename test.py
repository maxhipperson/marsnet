import os
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from nets import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data_dir = 'data.nosync/hs_data/mawrth_vallis/wl_None-2600/no_crop/preprocess_None'

    param_dir = 'data.nosync/nn/Net1'

    params = 'params_3.json'

    model = 'model_15.pt'

    param_path = os.path.join(param_dir, params)
    model_path = os.path.join(param_path[:-5], model)

    data_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]

    data = {}
    for data_file in data_files:
        array = np.load(os.path.join(data_dir, data_file))
        data[data_file[:-4]] = array
        print(data_file[:-4], array.shape)

    with open(param_path, 'r') as file:
        params = json.load(file)

    # Plot mean image

    fig, ax = plt.subplots(figsize=(10, 6))
    # fig.suptitle('')

    mean_im = np.mean(data['signal'], axis=2)
    ax.set_title('Mean Image', fontsize='medium')
    ax.set_ylabel('pixels / px')
    ax.set_xlabel('pixels / px')
    ax.imshow(mean_im)
    plt.show()

    model = Net1  # todo <--

    #########################
    # Make dataset iterable #
    #########################

    batch_size = params['training']['batch_size']

    spectra_arr = data['spectra_arr']
    dataset = TensorDataset(torch.Tensor(spectra_arr), torch.Tensor(spectra_arr))
    loader = DataLoader(dataset, batch_size)

    # Instantiate model

    input_shape = data['wavelengths'].shape[0]

    model = model(input_shape=input_shape, **params['model'])

    model.load_state_dict(torch.load(model_path))

    # for name, param in model.named_parameters():
    #     print(n)

    weights = model.encoder[0].weight
    weights = weights.detach().numpy()

    print(weights.shape)

    # Plot learned spectra

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title('Weights', fontsize='medium')
    for i in range(weights.shape[0]):
        ax.plot(data['wavelengths'] / 1000, weights[i], linewidth=1,  label='neuron {}'.format(i))
    ax.set_ylabel('')
    ax.set_xlabel(r'Wavelength / $\mu$m')
    fig.legend(fontsize='small')

    plt.show()

    #############
    # Run model #
    #############

    print()
    print('Running model')

    with torch.no_grad():
        model.eval()

        num_iters = len(loader)

        for iter, (x, _) in enumerate(loader):
            # Forward pass to get output
            output = model(x)
            encoded = model.encode(x)

            print(encoded.numpy()[0])

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.set_title('', fontsize='medium')
            ax.plot(data['wavelengths'] / 1000, x.numpy()[0], linewidth=1, label='input'.format(0))
            ax.plot(data['wavelengths'] / 1000, output.numpy()[0], linewidth=1, label='output'.format(0))
            ax.set_ylabel('')
            ax.set_xlabel(r'Wavelength / $\mu$m')
            fig.legend(fontsize='small')

            plt.show()

        # print('Epoch: {} / {} - loss: {:f} - val_loss: {:f}'.format(epoch, num_epochs, loss, val_loss))
