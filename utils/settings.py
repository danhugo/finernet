import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random

from configs import configs as cfg


plt.ion() # interactive mode for matplotlib
torch.backends.cudnn.benchmark  = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_VISIBLE_DEVICES

seed = cfg.SYSTEM.SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)