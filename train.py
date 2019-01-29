import os
import json
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from nets import *
from tensorboardX import SummaryWriter
from tqdm import tqdm


def cosine_loss(x, y):

    out = 1 - F.cosine_similarity(x, y)
    return torch.mean(out)


def record(value, metric, writer, valuename, writerstep):

    if metric is not None:
        metric.append(value)

    if writer is not None:
        writer.add_scalar(valuename, value, writerstep)


def run_epoch(epoch, loader, model, optimizer, criterion, writer, mode='train'):

    run_loss = []

    num_iters = len(loader)

    for iter, (x, _) in enumerate(tqdm(loader)):

        if mode == 'train':
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

        # Forward pass to get output
        outputs = model(x)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, x)

        if mode == 'train':
            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

        # Add iteration loss to metric and record
        record(loss.item(), run_loss, writer, 'iter/loss', num_iters * epoch + iter)

    run_loss = np.average(np.array(run_loss))

    # Add loss to metric and record
    record(run_loss, None, writer, 'loss', epoch)

    return run_loss


def run_model(params, train_data, test_data, input_shape, model, writerpath):

    # Set manual seed for reproducibility e.g. when shuffling the data
    torch.manual_seed(23)

    # Make dataset iterable

    batch_size = params['training']['batch_size']
    num_epochs = params['training']['num_epochs']

    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_data))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_data))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    # Instantiate model
    model = model(input_shape=input_shape, **params['model'])

    # Instantiate loss
    # criterion = torch.nn.MSELoss()  # todo <--
    criterion = cosine_loss

    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), **params['optimizer'])

    # Run model
    train_writer = SummaryWriter(writerpath)
    test_writer = SummaryWriter(writerpath + '_val')

    for epoch in range(num_epochs):

        model.train()

        # Train
        loss = run_epoch(epoch=epoch, loader=train_loader, model=model, optimizer=optimizer, criterion=criterion,
                         writer=train_writer, mode='train')

        # Test
        with torch.no_grad():
            model.eval()

            val_loss = run_epoch(epoch=epoch, loader=test_loader, model=model, optimizer=optimizer, criterion=criterion,
                                 writer=test_writer, mode='test')

        print('Epoch: {} / {} - loss: {:f} - val_loss: {:f}'.format(epoch, num_epochs, loss, val_loss))

        # Save model
        if epoch % 5 == 0:  # todo <-- comment out to prevent saving
            savepath = '{}/model_{}.pt'.format(os.path.split(writerpath)[0], epoch)
            torch.save(model.state_dict(), savepath)
            print('Saved {}'.format(savepath))


if __name__ == '__main__':

    data_dir = 'data.nosync/hs_data/mawrth_vallis/wl_None-2600/no_crop/preprocess_None'
    param_dir = 'data.nosync/nn/Net1'  # todo <--
    log_dir = 'data.nosync/nn_logs/basic'  # todo <--

    data_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]
    param_files = [f for f in sorted(os.listdir(param_dir)) if f.endswith('.json')]
    # param_files = ['paramset_man']

    data = {}
    for data_file in data_files:
        print(data_file)
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

        print()
        print('Param file: {}'.format(param_file))
        print()

        with open(os.path.join(param_dir, param_file), 'r') as file:
            params = json.load(file)

        for param, value in params.items():
            print('{}:\t{}'.format(param, value))

        writerpath = os.path.join(param_dir, param_file[:-5], 'log')

        # print()
        # print('Training  model with params:')
        # for key, value in params.items():
        #     writer = writer + '_{}'.format(value)
        #     print(key, value)

        model = Net1  # todo <--

        run_model(params,
                  train_data=train_data,
                  test_data=test_data,
                  input_shape=data['wavelengths'].shape[0],
                  model=model,
                  writerpath=writerpath)

