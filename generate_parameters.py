import numpy as np
import math
import random
import json
import os

# random.seed(23)


def lognuniform(base, low, high, size=None):
    return base ** np.random.uniform(math.log(low, base), math.log(high, base), size)


num_files = 9
dst_dir = 'data.nosync/nn_params/basic_autoencoder_dropout'

try:
    os.makedirs(dst_dir)
except OSError:
    pass

param_limits = {
    'num_epochs': 100,
    'n_endmembers': 5,
    # 'n_endmembers': 4,
    # 'n_endmembers': 5,
    # 'n_endmembers': [3, 4, 5],
    # 'n_features': [3, 100],
    # 'batch_size': [2**x for x in range(3, 7)],
    'dropout_p': [0.1, 0.9],
    'batch_size': [8, 256],
    # 'learning_rate': list(np.logspace(np.log10(1e-4), np.log10(1e-1), base=10, num=4)),
    'learning_rate': [1e-4, 1e-1],
}

for i in range(num_files):

    param_set = {
        'num_epochs': param_limits['num_epochs'],
        'n_endmembers': param_limits['n_endmembers'],
        # 'n_endmembers': random.sample(param_limits['n_endmembers'], 1)[0],
        # 'n_features': int(lognuniform(2, param_limits['n_features'][0], param_limits['n_features'][1]))
        'dropout_p': np.random.uniform(param_limits['dropout_p'][0], param_limits['dropout_p'][1]),
        'batch_size': int(lognuniform(10, param_limits['batch_size'][0], param_limits['batch_size'][1])),
        # 'batch_size': np.random.randint(param_limits['batch_size'][0], param_limits['batch_size'][1]),
        'learning_rate': lognuniform(10, param_limits['learning_rate'][0], param_limits['learning_rate'][1])
    }

    print()
    for key, value in param_set.items():
        print(key, value)

    savename = 'paramset_{}'.format(i)
    savepath = os.path.join(dst_dir, savename)

    with open(savepath, 'w') as file:
        json.dump(param_set, file, indent=0)
    print('Saved {}'.format(savepath))
