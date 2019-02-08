import numpy as np
import math
import random
import json
import os
import config as cfg

# random.seed(23)


def lognuniform(base, low, high, size=None):
    return base ** np.random.uniform(math.log(low, base), math.log(high, base), size)


num_files = 1
dst_dir = cfg.PARAM_DIR
# dst_dir = os.path.join('data.nosync', 'nn', 'Net1')

try:
    os.makedirs(dst_dir)
except OSError:
    pass

lim = {}

lim['training'] = {
    'num_epochs': 100,
    'batch_size': [4, 128],
}

lim['optimizer'] = {
    'lr': [1e-4, 1e-1],
}

lim['model'] = {
    'n_l1': [10, 100],
    'n_l2': [10, 100],
    'n_endmembers': [3, 4, 5],
    'dropout_p': [0.1, 0.9],

}

# Create parameter combinations

params = {
    'training': {},
    'optimizer': {},
    'model': {}
}

for i in range(num_files):

    params['training'] = {
        'num_epochs': lim['training']['num_epochs'],
        'batch_size': int(lognuniform(10, lim['training']['batch_size'][0], lim['training']['batch_size'][1])),
    }

    params['optimizer'] = {
        'lr': lognuniform(10, lim['optimizer']['lr'][0], lim['optimizer']['lr'][1])
    }

    params['model'] = {
        'n_l1': int(lognuniform(10, lim['model']['n_l1'][0], lim['model']['n_l1'][1])),
        'n_l2': int(lognuniform(10, lim['model']['n_l2'][0], lim['model']['n_l2'][1])),
        'n_endmembers': random.sample(lim['model']['n_endmembers'], 1)[0],
        'dropout_p': np.random.uniform(lim['model']['dropout_p'][0], lim['model']['dropout_p'][1]),
    }

    print()
    for key, value in params.items():
        print(key, value)

    savename = 'params_{}.json'.format(i)
    savepath = os.path.join(dst_dir, savename)

    with open(savepath, 'w') as file:
        json.dump(params, file, indent=4)

    print()
    print('Saved {}'.format(savepath))
