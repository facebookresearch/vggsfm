import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import hydra
from depth_fuser import DepthFuser
from torch.cuda.amp import autocast





fuser = torch.load("depth_fuserv2.pth")



# with autocast(dtype=torch.bfloat16):
fuser.prepare_for_opt()
fuser.optimize_depth()



import pdb;pdb.set_trace()
m=1
