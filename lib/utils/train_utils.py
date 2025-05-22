import os
import torch
import random
import numpy as np


def worker_init_fn(worder_id):
    np.random.seed(np.random.get_state()[1][0] + worder_id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False