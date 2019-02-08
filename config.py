import os
from os.path import join
from nets import *

# IMG_EXTENSIONS = ('.jpg', '.png', '.PNG', '.JPG', '.JPEG', '.jpeg')

# DIRECTORIES
##################################################

PROJECT_DIR = os.path.expanduser('~/Documents/Year 4/project/marsnet')

PROJECT_DATA_DIR = join(PROJECT_DIR, 'data.nosync')




# TRAINING
##################################################

MODEL = Net1
# MODEL = Net1Dropout

# MODEL = Net2
# MODEL = Net3

# DATA_DIR = join(PROJECT_DATA_DIR, 'hs_data', 'mawrth_vallis', 'wl_None-2600', 'preprocess_None')
DATA_DIR = join(PROJECT_DATA_DIR, 'hs_data', 'mawrth_vallis', 'wl_1000-2600', 'preprocess_None')

PARAM_DIR = join(PROJECT_DATA_DIR, 'runs', 'test')
# PARAM_DIR = join(PROJECT_DATA_DIR, 'runs', 'test2')
