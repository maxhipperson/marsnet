import numpy as np
import os


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

        print()
        print('{}: {}'.format(param, value))

    return params
