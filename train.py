import os
import json
import numpy as np
import config as cfg
import csv

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from nets import *
from tensorboardX import SummaryWriter


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


def get_train_test_split(data, test_frac=0.1):

    temp = data.copy()
    np.random.shuffle(temp)

    idx = int(temp.shape[0] * test_frac)

    train_data = temp[:idx]
    test_data = temp[idx:]

    return train_data, test_data


def load_params(param_dir, param_file):

    print()
    print('=' * 50)
    print('Loaded {}'.format(param_file))

    with open(os.path.join(param_dir, param_file), 'r') as file:
        params = json.load(file)

    for param, value in params.items():

        print()
        print('{}: {}'.format(param, value))

    return params


def cosine_loss(x, y):

    out = 1 - F.cosine_similarity(x, y)
    return torch.mean(out)


def record(value, metric, writer, valuename, writerstep):

    if metric is not None:
        metric.append(value)

    if writer is not None:
        writer.add_scalar(valuename, value, writerstep)


def train(params, train_data, test_data, input_shape, model, writerpath):

    # Set manual seed for reproducibility e.g. when shuffling the data
    torch.manual_seed(0)

    # Make dataset iterable

    batch_size = params['training']['batch_size']
    num_epochs = params['training']['num_epochs']

    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_data))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_data))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    # Instantiate model
    model = model(input_shape=input_shape, **params['model'])

    print()
    print(model)

    # Instantiate loss
    # criterion = torch.nn.MSELoss()  # todo <--
    criterion = cosine_loss

    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), **params['optimizer'])  # todo - learning rate decay?


    # Run model
    train_writer = SummaryWriter(os.path.join(writerpath, 'train'))
    test_writer = SummaryWriter(os.path.join(writerpath, 'val'))

    for epoch in range(num_epochs):

        model.train()

        # Train
        ##################################################

        train_loss = []

        num_iters = len(train_loader)

        for iter, (x, _) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output
            __, out = model(x)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(out, x)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            model.encoder[0].weight.data.clamp_(0, 1)

            # Add iteration loss to metric and record
            record(loss.item(), train_loss, train_writer, 'iter/train_loss', num_iters * epoch + iter)
            # record(loss.item(), train_loss, None, 'iter/loss', num_iters * epoch + iter)

        train_loss = np.average(np.array(train_loss))

        # Add loss to metric and record
        record(train_loss, None, train_writer, 'loss', epoch)

        # Val
        ##################################################

        with torch.no_grad():

            model.eval()

            test_loss = []

            num_iters = len(test_loader)

            for iter, (x, _) in enumerate(test_loader):

                # Forward pass to get output
                __, out = model(x)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(out, x)

                # Add iteration loss to metric and record
                record(loss.item(), test_loss, test_writer, 'iter/test_loss', num_iters * epoch + iter)
                # record(loss.item(), test_loss, None, 'iter/val_loss', num_iters * epoch + iter)

            test_loss = np.average(np.array(test_loss))

            # Add loss to metric and record
            record(test_loss, None, test_writer, 'loss', epoch)

        print('epoch: {} / {} - loss: {:f} - val_loss: {:f}'.format(epoch, num_epochs, train_loss, test_loss))

        # Write the metrics to text log
        with open(os.path.join(writerpath, 'log.csv'), 'a') as csvfile:  # todo

            csvwriter = csv.writer(csvfile)

            if epoch == 0:

                fieldnames = ['epoch', 'loss', 'val_loss']
                csvwriter.writerow(fieldnames)

            csvwriter.writerow([epoch, train_loss, test_loss])

        # Save model
        if epoch % 5 == 0:  # todo <-- comment out to prevent saving
            savepath = os.path.join(writerpath, 'model_epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), savepath)
            print('Saved {}'.format(savepath))


if __name__ == '__main__':

    data_dir = cfg.DATA_DIR
    param_dir = cfg.PARAM_DIR

    data_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]
    param_files = [f for f in sorted(os.listdir(param_dir)) if f.endswith('.json')]

    data = load_data(data_dir, data_files)

    np.random.seed(0)

    # Split the spectra into a train and test set
    train_data, test_data = get_train_test_split(data['spectra_arr'])

    # Load each parameter file and train the model
    for param_file in param_files:

        params = load_params(param_dir, param_file)

        writerpath = os.path.join(param_dir, param_file[:-5])

        model = cfg.MODEL

        print()
        print('-' * 50)
        print('Loaded model: {}'.format(model.__name__))  # todo get name from class

        print()
        print('-' * 50)
        print('Begin training')

        train(params,
              train_data=train_data,
              test_data=test_data,
              input_shape=data['wavelengths'].shape[0],
              model=model,
              writerpath=writerpath)

