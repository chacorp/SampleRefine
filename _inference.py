'''
after loading pytorch_test docker, pip install piq matplotlib lpips scipy
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np

# import time
import torch
import torch.utils.data

import torchvision.transforms.functional as F

from utils.train_options import TrainOptions
from utils.dataloader import img2tensor

from sample_refine import Model
from tqdm import tqdm
from glob import glob

def is_bad_grad(variable):
    return 'has grad' if variable.requires_grad == True else 'no grad'

def set_seed(seed):
    from torch.backends import cudnn
    import random
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_image_files_from_path(path):
    if path != None:
        if ".jpg" in path or ".png" in path:
            image_files = [path]
        else:    
            image_files = sorted(glob(os.path.join(path, "*.jpg")) + sorted(glob(os.path.join(path, "*.png"))))
    else:
        image_files = []
    return image_files

def main(opt):
    set_seed(1234)

    opt.mode        = 'test'
    #######[ Get Model ]#######################################################################
    opt.device_id   = [i for i in range(torch.cuda.device_count())]
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = Model(opt=opt, device=device)
    model.eval()
    
    # given a directory
    img_paths    = get_image_files_from_path(opt.input) #[opt.input]
    msk_paths    = get_image_files_from_path(opt.pmask) #[opt.pmask]
    nrm_paths    = get_image_files_from_path(opt.norm_map) #[opt.norm_map]

    pbar = tqdm(zip(img_paths, msk_paths, nrm_paths))
    for img_file, msk_file, nrm_file in pbar:
        pbar.set_description(f"{img_file}")
        
        data = {
            'T_inputA' :  img2tensor(img_file, resize=True, size=opt.data_size)[None],
            'T_inputB' :  img2tensor(img_file, resize=True, size=opt.data_size)[None],
            'Vis_maskA' :  img2tensor(msk_file, resize=True, size=opt.data_size)[0][None][None],
            'Vis_maskB' :  img2tensor(msk_file, resize=True, size=opt.data_size)[0][None][None],
            'GT_texture':  torch.zeros([]),
            'norm_map': img2tensor(nrm_file, resize=True, size=opt.data_size)[None],
            'has_GT': False,
            'sampled_imageA':img2tensor(img_file, resize=True, size=opt.data_size)[None], # for RefinerNet
        }
        with torch.no_grad():
            s_inputA = torch.cat((
                    data['T_inputA'].to(device), 
                    data['norm_map'].to(device),
                    data['Vis_maskA'].to(device)
                ), dim=1)
            sampled_texture = model.sampler(s_inputA, mode='s')
            data['sampled_imageA'] = sampled_texture.detach().requires_grad_()

        prev_out, output, _, _ = model(data, mode='visualize')

        name = img_file.split('/')[-1][:-13] # 13 = len('_symmetry.png')
        if opt.G in ['S1', 'S2']:
            prev_name = 'partial'
        else:
            prev_name = 'sampled'
        F.to_pil_image(prev_out[0]).save('{}/{}_{}.png'.format(opt.out, name, prev_name))
        F.to_pil_image(output[0]).save('{}/{}_output.png'.format(opt.out, name))

if __name__ == "__main__":
    train_opt = TrainOptions()
    opt = train_opt.parse()
    os.makedirs(opt.out, exist_ok=True)
    main(opt=opt)