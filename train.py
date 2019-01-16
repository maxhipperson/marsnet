import os
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from nets import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

def train(params, train_data, test_data, input_shape, model, writername):

    # Set manual seed for reproducibility e.g. when shuffling the data
    torch.manual_seed(23)

    #########################
    # Make dataset iterable #
    #########################

    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_data))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_data))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    #####################
    # Instantiate model #
    #####################

    input_shape = input_shape
    n_endmembers = params['n_endmembers']

    model = model(input_shape, n_endmembers)

    # Set model to train mode
    model.train()

    ####################
    # Instantiate loss #
    ####################

    criterion = torch.nn.MSELoss()

    #########################
    # iIstantiate optimizer #
    #########################

    learning_rate = params['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #########
    # Train #
    #########

    num_iters_train = len(train_loader)
    num_iters_test = len(test_loader)

    writer = SummaryWriter(writername)

    for epoch in range(num_epochs):

        for iter, (spectra, _) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output
            outputs = model(spectra)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, spectra)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            writer.add_scalar('train/loss/iter', loss.item(), num_iters_train * epoch + iter)
            # print('\t iter {} / {} - loss {}'.format(iter, num_iters, loss.data.item()))

        writer.add_scalar('train/loss/epoch', loss.item(), epoch)
        # print('epoch {} / {} - train_loss {:f}'.format(epoch + 1, num_epochs, loss.item()))

        #########
        # Test #
        #########

        with torch.no_grad():

            model.eval()

            for iter, (spectra, _) in enumerate(test_loader):

                # Forward pass to get output
                outputs = model(spectra)

                # Calculate Loss: softmax --> cross entropy loss
                test_loss = criterion(outputs, spectra)

                writer.add_scalar('test/loss/iter', test_loss.item(), num_iters_test * epoch + iter)

            writer.add_scalar('test/loss/epoch', test_loss.item(), epoch)
            print('epoch {} / {} - loss: train {:f} - test {:f}'.format(epoch + 1, num_epochs, loss.item(), test_loss.item()))


if __name__ == '__main__':

    data_dir = 'data.nosync/hs_data/mawrth_vallis/wl_None-2600/no_crop/preprocess_None'
    param_dir = 'data.nosync/nn_params/basic_autoencoder'
    log_dir = 'data.nosync/nn_logs/basic_autoencoder'

    data_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]
    param_files = [f for f in sorted(os.listdir(param_dir))]

    data = {}
    for data_file in data_files:
        array = np.load(os.path.join(data_dir, data_file))
        data[data_file[:-4]] = array
        print(data_file[:-4], array.shape)

    # Split the spectra into a train and test set
    np.random.seed(23)

    temp = data['spectra_arr']
    np.random.shuffle(temp)

    idx = int(temp.shape[0] * 0.9)

    train_data = temp[:idx]
    test_data = temp[idx:]

    print()
    print('n_samples: {}'.format(data['spectra_arr'].shape[0]))
    print('train n_samples: {}'.format(train_data.shape[0]))
    print('test n_samples: {}'.format(test_data.shape[0]))

    # Load each parameter file and train the model
    for param_file in param_files:

        with open(os.path.join(param_dir, param_file), 'r') as file:
            params = json.load(file)

        writer = log_dir + '/' + param_file

        # print()
        # print('Training  model with params:')
        # for key, value in params.items():
        #     writer = writer + '_{}'.format(value)
        #     print(key, value)

        model = Basic

        train(params,
              train_data=train_data,
              test_data=test_data,
              input_shape=data['wavelengths'].shape[0],
              model=model,
              writername=writer)

