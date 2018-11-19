import torch
import tensorboardX
import time
import config as cfg
import os

use_gpu = torch.cuda.is_available()

time_stamp = time.strftime('%y-%m-%d.%H-%M-%S')
run_id = str(cfg.MODEL_NAME) + '.' + time_stamp
run_log_dir = os.path.join(cfg.LOG_DIR, 'active', run_id)

