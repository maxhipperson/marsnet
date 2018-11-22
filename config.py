import os
from os.path import join

# IMG_EXTENSIONS = ('.jpg', '.png', '.PNG', '.JPG', '.JPEG', '.jpeg')

### DIRECTORIES ###

PROJECT_DIR = os.path.expanduser('~/Documents/Year 4/marsnet')
DATA_DIR = join(PROJECT_DIR, 'data.nosync')
LOG_DIR = join(DATA_DIR, 'logs')

SITE_DIR = join(DATA_DIR, '')
# SITE_DIR = join(DATA_DIR, '')
# SITE_DIR = join(DATA_DIR, '')

### TRAINING ###

MODEL = ''
MODEL_NAME = ''

PARAMS = {}

NUM_EPOCHS = 1
NUM_ITERS = 1
