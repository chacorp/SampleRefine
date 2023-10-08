# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from os.path import join
from glob import glob
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import kornia.augmentation as K

from utils.utils import set_seed
# augmentation
from utils.transformation import (
    tps_transform_kornia,
    affine_transform,
    rotate_transform,
    sheer_transform,
    translate_transform,
    naive_tps_transform_kornia
)
import random

def _get_coords(h, w, ds, bs=1):
    """Creates the position encoding for the pixel-wise MLPs"""
    from math import pi
    from math import log2
    f0 = ds
    f = f0
    while f > 1:
        x = torch.arange(0, w).float()
        y = torch.arange(0, h).float()
        xcos = torch.cos((2 * pi * torch.remainder(x, f).float() / f).float())
        xsin = torch.sin((2 * pi * torch.remainder(x, f).float() / f).float())
        ycos = torch.cos((2 * pi * torch.remainder(y, f).float() / f).float())
        ysin = torch.sin((2 * pi * torch.remainder(y, f).float() / f).float())
        xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
        xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
        ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
        ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
        coords_cur = torch.cat([xcos, xsin, ycos, ysin], 1)
        if f < f0:
            coords = torch.cat([coords, coords_cur], 1)
        else:
            coords = coords_cur
        f = f//2

    return coords.squeeze(0)

def load_npy(file):
    with open(file, 'rb') as f:
        return np.load(f)

def img2tensor(path, resize=False, size=512, resample=Image.LANCZOS): # Resampling
    input = Image.open(path)
    if resize:
        input = input.resize((size, size), resample=resample)
    input = transforms.ToTensor()(input)[:3]
    return input

def img2numpy(path, resize=False, size=512):
    input = Image.open(path)
    if resize:
        input = input.resize((size, size), resample=Image.LANCZOS) # Resampling
    input = np.array(input)
    return input

def _get_mask(path='./utils/_mask.png', resize=False, size=256, bn=True):
    mask = img2tensor(path, resize=resize, size=size)
    mask = (mask > 0).all(0) * 1.0
    if bn == False:
        mask = mask * 1.0
    return mask

def _get_part_mask(resize=False, size=256):
    uv_mask = _get_mask(resize=resize, size=size)

    face = _get_mask('./utils/uv_masks/face.png',resize=resize, size=size)
    body = _get_mask('./utils/uv_masks/body.png',resize=resize, size=size)
    legs = _get_mask('./utils/uv_masks/both_leg.png',resize=resize, size=size)
    arms = _get_mask('./utils/uv_masks/both_arm.png',resize=resize, size=size)
    foot = _get_mask('./utils/uv_masks/foot.png',resize=resize, size=size)
    hand = _get_mask('./utils/uv_masks/hand.png',resize=resize, size=size)

    mask = torch.stack((face, body, legs, arms, foot, hand), dim=0)

    mask = mask * uv_mask
    return mask # [6, 256, 256]

def _get_part_mask_bn(resize=False, size=256):
    uv_mask = _get_mask(resize=resize, size=size)

    face = _get_mask('./utils/uv_masks/face.png',resize=resize, size=size)
    body = _get_mask('./utils/uv_masks/body.png',resize=resize, size=size)
    legs = _get_mask('./utils/uv_masks/both_leg.png',resize=resize, size=size)
    arms = _get_mask('./utils/uv_masks/both_arm.png',resize=resize, size=size)
    foot = _get_mask('./utils/uv_masks/foot.png',resize=resize, size=size)
    hand = _get_mask('./utils/uv_masks/hand.png',resize=resize, size=size)

    body = (body * uv_mask) > 0
    face = (face * uv_mask) > 0
    legs = (legs * uv_mask) > 0
    arms = (arms * uv_mask) > 0
    foot = (foot * uv_mask) > 0
    hand = (hand * uv_mask) > 0

    mask = torch.stack((face, body, legs, arms, foot, hand), dim=0)
    return mask # [6, 256, 256]

def augment_color(img, p=0.8):
    aug = K.ColorJitter(
                brightness    = 0.01,
                contrast      = 0.01,
                saturation    = 0.5,
                hue           = 0.5,
                p             = p,
                same_onV2atch = True, 
    )
    return aug(img)

def augment_blur(img, p=0.8):
    if torch.rand(1) > p:
        aug = transforms.GaussianBlur(7, sigma=(0.1, 2.0))
        img = aug(img)
    return img

class ViewDataset(Dataset):
    def __init__(self, opt, log=None, test=False, rendered=False, angle=None):
        
        self.opt        = opt
        self.size       = opt.data_size         # (torch.Tensor) desired size of the texture
        self.resize     = opt.resize            # (bool) if true resize the data
        self.aug_color  = opt.aug_color         # (bool) if true augment color of the data
        self.interval   = opt.interval          # (int) interval of view in azimoth, must be multiples of 10
        
        if not self.opt.progressive:            # get transform function for the image augmentation
            if self.opt.tps:
                self.transform = tps_transform_kornia
            elif self.opt.naive_tps:
                self.transform = naive_tps_transform_kornia     
            elif self.opt.affine:
                self.transform = affine_transform
            elif self.opt.rotate:
                self.transform = rotate_transform
            elif self.opt.sheer:
                self.transform = sheer_transform
            elif self.opt.translate:
                self.transform = translate_transform
            else:
                self.transform = None    
        else:
            self.transform = None

        if test:
            file_name = "data_list_test.json"
        else:
            file_name = "data_list_train.json"

        import json
        with open(file_name, "r") as file:
            json_dict = json.load(file)

        self.data_list = json_dict['data_list']
        self.txtr_map  = json_dict['txtr_map']
        self.mask      = json_dict['mask']
        self.part_map  = json_dict['part_map']
        self.norm_map  = json_dict['norm_map']
        self.interval  = json_dict['interval']

        # default
        self.angle = np.arange(-90, 100, self.interval)
        
        self.len_angle    = len(self.angle) # 19 (-90~90)
        self.len_datasets = len(self.txtr_map) # 5
        self.len_datalist = len(self.data_list) # 5

        self.uv_mask = _get_mask(resize=self.resize,  size=self.size)
        self.label   = _get_part_mask(resize=self.resize, size=self.size)

        #import pdb;pdb.set_trace()

    def __len__(self):
        return sum([n for n, d in self.data_list]) * len(self.angle)
    
    def _get_key(self, index): # get key in data_list 
        num = 0
        for i in range(self.len_datalist): # 5
            dataset_size = self.data_list[i][0] # 300, 451, 478, 212, 300
            curr_num = num + (dataset_size * self.len_angle)
            if num <= index < curr_num:
                return i, num # ith dataset, num: data index?
            num = curr_num

    def _get_other_angle(self, a_idx):
        while 1: # to be V1 != V2
            b_idx = random.randint(0, 18)
            if a_idx != b_idx:
                return self.angle[b_idx]

    def __getitem__(self, index):
        """
        ## TODO: implement the code for returning the data as below
        # 1. get T_input (A) from certain angle
        # 2. get T_input (B) from another angle
        # 3. get corresponding normal map
        # 4. get corresponding GT texture and set 'has_GT' to True
        # 5. if not set 'has_GT' to False
        # 6. return data
        ### note that textures should be normalized between -1 ~ 1 // mask = 0 ~ 1
        """
        # 1. get dataset
        # _key: ith dataset(0,1,2,3,4) _prev_data_len: data index? length of previous data
        _key, _prev_data_len  = self._get_key(index)             # [0, 1, 2], length of previous data
        _datalen, _ = self.data_list[_key]       # dataset length, name

        _idx = (index - _prev_data_len) % _datalen        # selected data index
        
        # 2. get view angle (V1, V2)
        # V1 = self.angle[index % self.len_angle] # angle [-90, -80, -70, ..., 80, 90]
        V1_idx = (index - _prev_data_len) // _datalen

        V1 = str(self.angle[V1_idx])
        V2 = str(self._get_other_angle(V1_idx))
        _K = str(_key)
        # 3. target  mask  partial
        ########################### Bottle neck ############################
        # for angleA        
        anc_targetA   = img2tensor(self.txtr_map[_K][_idx],    resize=self.resize, size=self.size)
        anc_norm      = img2tensor(self.norm_map[_K][_idx],    resize=self.resize, size=self.size)        
        anc_maskA     = img2tensor(self.mask[_K][V1][_idx],    resize=self.resize, size=self.size)
        if not self.opt.progressive and self.opt.masking:
            anc_partA = anc_targetA * anc_maskA
        else:
            anc_partA = img2tensor(self.part_map[_K][V1][_idx],resize=self.resize, size=self.size)
            anc_partA = anc_partA * anc_maskA

        # for angleB
        anc_targetB   = img2tensor(self.txtr_map[_K][_idx],    resize=self.resize, size=self.size)
        anc_maskB     = img2tensor(self.mask[_K][V2][_idx],    resize=self.resize, size=self.size)
        if not self.opt.progressive and self.opt.masking:
            anc_partB = anc_targetB * anc_maskB
        else:
            anc_partB = img2tensor(self.part_map[_K][V2][_idx],resize=self.resize, size=self.size)
            anc_partB = anc_partB * anc_maskB 
        ########################### Bottle neck ############################

        # need to mask
        anc_targetA = anc_targetA * self.uv_mask
        anc_targetB = anc_targetB * self.uv_mask
        # 4. transformation
        if self.transform != None:
            CV1, MV1 = anc_partA.size(0), anc_maskA.size(0)
            CV2, MV2 = anc_partB.size(0), anc_maskB.size(0)
            tempA = torch.cat((anc_partA, anc_maskA), dim=0)[None]
            tempB = torch.cat((anc_partB, anc_maskB), dim=0)[None]
            tempA = self.transform(
                tempA, 
                p          = self.opt.p, 
                scale      = self.opt.tps_scale, 
                input_size = self.size,
                label_mask = self.label
            )[0]
            tempB = self.transform(
                tempB, 
                p          = self.opt.p, 
                scale      = self.opt.tps_scale, 
                input_size = self.size,
                label_mask = self.label
            )[0]
            anc_partA, anc_maskA = tempA[:CV1], tempA[CV1:]
            anc_partB, anc_maskB = tempB[:CV2], tempB[CV2:]
            
        # 5. apply augment (color)
        if self.aug_color:
            temp_imgA = torch.cat((anc_partA[None], anc_targetA[None]), dim=0)
            temp_imgB = torch.cat((anc_partB[None], anc_targetB[None]), dim=0)
            temp_imgAB = torch.cat([temp_imgA, temp_imgB]) # combine imgA, imgB
            
            augmentedV2atch = augment_color(temp_imgAB, p=self.opt.p) # imgA, imgB same augmentation 
            anc_partA, anc_targetA = augmentedV2atch[0], augmentedV2atch[1]
            anc_partB, anc_targetB = augmentedV2atch[2], augmentedV2atch[3]

        data = {
            'T_inputA'  : anc_partA,
            'T_inputB'  : anc_partB,
            'Vis_maskA' : anc_maskA,
            'Vis_maskB' : anc_maskB,
            'GT_texture': anc_targetA,
            'norm_map'  : anc_norm,
            'has_GT': True,
        }

        if self.opt.progressive or self.opt.masking:
            if self.transform != None and self.opt.masking:
                data['maskingA'] = anc_partA
                data['maskingB'] = anc_partB
            else:
                data['maskingA'] = anc_targetA * anc_maskA
                data['maskingB'] = anc_targetB * anc_maskB

        return data

class PairedDataset(Dataset):
    def __init__(self, opt, 
        log=None, test=False, rendered=False, angle=None):

        self.opt        = opt
        self.size       = opt.data_size
        self.resize     = opt.resize
        self.aug_color  = opt.aug_color
        self.interval   = opt.interval
        self.mirror     = opt.mirror
        self.naive_tps  = opt.naive_tps
        
        # ----------------------------------------- modify the path relative to your machine
        # self.data_path = '../TexturePaired'
        self.data_path = '/data/sihun/TexturePaired'
        # self.data_path = '/source/jihyeon/Sample-Refine2/TexturePaired/'
        # ----------------------------------------------------------------------------------
        
        if test:
            self.opt.tps, self.opt.affine, self.opt.naive_tps = False, False, False
            self.opt.rotate, self.opt.translate = False, False
            self.interval = 10
            
        self.rendered = opt.rendered_data if self.opt else rendered
        
        if not self.opt.progressive:
            if self.opt.tps:
                self.transform = tps_transform_kornia
            elif self.opt.naive_tps:
                self.transform = naive_tps_transform_kornia     
            elif self.opt.affine:
                self.transform = affine_transform
            elif self.opt.rotate:
                self.transform = rotate_transform
            elif self.opt.sheer:
                self.transform = sheer_transform
            elif self.opt.translate:
                self.transform = translate_transform
            else:
                self.transform = None    
        else:
            self.transform = None        
        
        if test:
            file_name = "data_list_test.json"
        else:
            file_name = "data_list_train.json"

        import json
        with open(file_name, "r") as file:
            json_dict = json.load(file)

        self.data_list = json_dict['data_list']
        self.txtr_map  = json_dict['txtr_map']
        self.mask      = json_dict['mask']
        self.part_map  = json_dict['part_map']
        self.norm_map  = json_dict['norm_map']
        self.interval  = json_dict['interval']

        self.angle = np.arange(-90, 100, self.interval)
        self.len_angle    = len(self.angle) # 19 (-90~90)
        self.len_datasets = len(self.txtr_map) # 5
        self.len_datalist = len(self.data_list) # 5
        self.uv_mask = _get_mask(resize=self.resize,  size=self.size)
        self.label   = _get_part_mask(resize=self.resize, size=self.size)

    def __len__(self):
        return sum([n for n, d in self.data_list]) * len(self.angle)

    def _get_key(self, index): # get key in data_list 
        num = 0
        for i in range(self.len_datalist): # 5
            dataset_size = self.data_list[i][0] # 300, 451, 478, 212, 300
            curr_num = num + (dataset_size * self.len_angle)
            if num <= index < curr_num:
                return i, num # ith dataset, num: data index?
            num = curr_num

    def __getitem__(self, index):
        # 1. get dataset
        # _key  = np.random.randint(self.len_datasets)  # [0, 1, 2]
        # _idx = np.random.randint(_datalen)      # number
        # _key: ith dataset(0,1,2,3,4) _prev_data_len: data index? length of previous data
        _key, _prev_data_len  = self._get_key(index)             # [0, 1, 2], length of previous data
        _datalen, _dataset = self.data_list[_key]       # dataset length, name
        _idx = (index - _prev_data_len) % _datalen # data select?
        
        # 2. get angle
        # V1ngle = self.angle[index % self.len_angle] # angle [-90, -80, -70, ..., 80, 90]
        V1_idx = (index - _prev_data_len) // _datalen
        V1 = str(self.angle[V1_idx])
        _K = str(_key)

        # 3. target  mask  partial
        ########################### Bottle neck ############################
        # for angle
        anc_target = img2tensor(self.txtr_map[_K][_idx],     resize=self.resize, size=self.size)
        anc_norm   = img2tensor(self.norm_map[_K][_idx],     resize=self.resize, size=self.size)        
        anc_mask   = img2tensor(self.mask[_K][V1][_idx],     resize=self.resize, size=self.size)
        
        if not self.opt.progressive and self.opt.masking:
            anc_part = anc_target * anc_mask
        else:
            anc_part = img2tensor(self.part_map[_K][V1][_idx],resize=self.resize, size=self.size)
            anc_part = anc_part * anc_mask
       
        ########################### Bottle neck ############################

        # need to mask
        anc_target = anc_target * self.uv_mask
        # 4. transformation
        if self.transform != None:
            C, M = anc_part.size(0), anc_mask.size(0)
            temp = torch.cat((anc_part, anc_mask), dim=0)[None]
            temp = self.transform(
                temp, 
                p          = self.opt.p, 
                scale      = self.opt.tps_scale, 
                input_size = self.size,
                label_mask = self.label
            )[0]
            anc_part, anc_mask = temp[:C], temp[C:]
            
        # 5. apply augment (color)
        if self.aug_color:
            temp_img = torch.cat((anc_part[None], anc_target[None]), dim=0)
            temp_img = augment_color(temp_img, p=self.opt.p)
            anc_part, anc_target = temp_img[0], temp_img[1]

        data = {
            'T_inputA'  : anc_part,
            'T_inputB'  : anc_part,
            'Vis_maskA' : anc_mask,
            'Vis_maskB' : anc_mask,
            'GT_texture': anc_target,
            'norm_map'  : anc_norm,
            'has_GT': True,
            # 'norm_map': anc_norm * self.uv_mask,
        }

        if self.opt.progressive or self.opt.masking:
            if self.transform != None and self.opt.masking:
                data['masking'] = anc_part
            else:
                data['masking'] = anc_target * anc_mask

        return data


def test():
    set_seed(123456)
    from torchvision.transforms import ToPILImage
    from torch.utils.data import DataLoader
    from utils.dataloader import PairedDataset as CustomDataset
    from utils.dataloader import ViewDataset as ViewDataset
    from utils.train_options import TrainOptions
    train_opt = TrainOptions()
    opt = train_opt.parse()
    # opt.mirror = True
    # opt.masking = True

    # opt.sheer = True
    # opt.tps_scale = 0.25
    # opt.p = 0.999

    # opt.affine = True
    opt.aug_color = True
    # dataset = CustomDataset(opt, angle=0)
    dataset = ViewDataset(opt, angle=0)
    if 0:
        data    = next(iter(dataset))
        print(f'dataset: {len(dataset)}')
        for k in data.keys():
            ToPILImage()(data[k]).save('test_pd_{}.png'.format(k))
    else:
        dataset = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
        data    = next(iter(dataset))
        print(f'dataset: {len(dataset)}')
        for k in data.keys(): 
            if k == 'has_GT': # has_GT except
                break
            ToPILImage()(data[k][0]).save('test_pd_{}.png'.format(k))
"""
    python -c "from utils.dataloader import test; test()"
"""
if __name__ == "__main__":
    test()