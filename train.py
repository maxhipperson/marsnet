import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from nets import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

# Set manual seed for reproducibility e.g. when shuffling the data
torch.manual_seed(23)

#############
# Load data #
#############

src_dir = 'data.nosync/hs_data/mawrth_vallis/wl_None-2600/no_crop/preprocess_None'

files = [f for f in sorted(os.listdir(src_dir)) if f.endswith('.npy')]

data = {}
print('Loading data from {}:'.format(src_dir))
for file in files:
    array = np.load(os.path.join(src_dir, file))
    data[file[:-4]] = array
    print('{} - {}'.format(file[:-4], array.shape))

#########################
# Make dataset iterable #
#########################

batch_size = 16
num_epochs = 10

train_dataset = TensorDataset(torch.Tensor(data['spectra_arr']), torch.Tensor(data['spectra_arr']))

# train_loader = DataLoader(train_dataset, batch_size)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

#####################
# Instantiate model #
#####################

input_shape = data['wavelengths'].shape[0]
n_endmembers = 5  # todo <--

model_name = 'simple'
model = Simple(input_shape, n_endmembers)

####################
# Instantiate loss #
####################

criterion = torch.nn.MSELoss()

#########################
# iIstantiate optimizer #
#########################

# learning_rate = 1e-1
# learning_rate = 1e-2
learning_rate = 1e-3
# learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#########
# Train #
#########

num_iters = len(train_loader)

# writer = SummaryWriter('logs/{}_{}'.format(batch_size, learning_rate))
writer = SummaryWriter('data.nosync/logs/{}/{}_{}_{}_shuffle'.format(model_name, n_endmembers, batch_size, learning_rate))

for epoch in range(num_epochs):
    for iter, (spectra, _) in enumerate(train_loader):

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(spectra)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, spectra)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        if iter == 0 and epoch == 0:
            initial_loss = loss.item()

        writer.add_scalar('loss_iter', loss.item(), num_iters * epoch + iter)

        # print('\t iter {} / {} - loss {}'.format(iter, num_iters, loss.data.item()))

    writer.add_scalar('loss_epoch', loss.item(), epoch)
    print('epoch {} / {} - loss {}'.format(epoch + 1, num_epochs, loss.item()))
