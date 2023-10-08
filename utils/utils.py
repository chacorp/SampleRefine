# from __future__ import division
from builtins import ValueError
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm

import cv2
import PIL.Image as Image
import numpy as np
import torchvision.utils as tvu

def set_seed(seed):
    from torch.backends import cudnn
    import random
    cudnn.benchmark     = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_bad_grad(variable):
    return 'has grad' if variable.requires_grad == True else 'no grad'

def save_feat(feature, opt, type='enc', num=0, emp=False, gc=False):
    if opt.use_gate or gc:
        feature, gate = feature
    
    feat = feature.mean(1).unsqueeze(1)
    if emp:
        feat = feat * 4
    
    tvu.save_image(feat, f"{opt.out}/{type}_{str(num).zfill(3)}.png", nrow=1, padding=0)#, normalize=True, range=(-1,1))

    if opt.use_gate or gc:
        mask = gate.mean(1).unsqueeze(1)
        tvu.save_image(mask, f"{opt.out}/{type}_gate_{str(num).zfill(3)}.png", nrow=1, padding=0)
        feature = feature * gate

    return feature
