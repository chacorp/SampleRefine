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
            'norm_map':img2tensor(nrm_file, resize=True, size=opt.data_size)[None],
            'has_GT': True,
            # 'samp_map':torch.zeros([]),
            'sampled_imageA':img2tensor(img_file, resize=True, size=opt.data_size)[None], # for rTG
        }

        sampled, predicted, _, _ = model(data, mode='visualize')

        name = img_file.split('/')[-1][:-13] # 13 = len('_symmetry.png')
        F.to_pil_image(sampled[0]).save('{}/{}_sampled.png'.format(opt.out, name))
        F.to_pil_image(predicted[0]).save('{}/{}_output.png'.format(opt.out, name))

if __name__ == "__main__":
    train_opt = TrainOptions()
    opt = train_opt.parse()
    opt.mode       = 'test'
    opt.G          = 'R1'
    opt.SamplerNet = 'norm_map_vis_mask'
    opt.concat     = 'vis_mask_c'
    opt.use_gate   = True
    opt.mirror     = True
    opt.num_blocks = 9
    opt.num_layers = 3
    opt.Refine_mode == 'blend'

    os.makedirs(opt.out, exist_ok=True)
    
    ### from dataset # generate T_sample
    main(opt=opt)

"""
## EG 2023 version
python _inference.py \
	--input demo/demo_input/tex_1_angle_10005_symmetry.png \
	--pmask demo/demo_input/tex_1_angle_10005_mask_symm.png \
	--norm_map demo/demo_input/tex_1_angle_10005_norm.png \
 	--out infer \
 	--checkpoint Refiner-mirror-vis_mask_c \
 	--usermemo Inference --infer
  
python _inference.py \
	--input /data/sihun/Sample-Refine2/infer/test/rendered_uv5.png \
	--pmask /data/sihun/Sample-Refine2/infer/test/rendered_vis_mask3.png \
	--norm_map /data/sihun/Sample-Refine2/infer/test/T_M_012_001_norm.png \
 	--out infer/test \
 	--checkpoint Refiner-mirror-vis_mask_c4 \
 	--usermemo Inference --infer
  
python _inference.py \
    --input /source/Sample-Refine/demo/demo_image/symmetry/image \
    --pmask /source/Sample-Refine/demo/demo_image/mask_symm/image \
    --norm_map /source/Sample-Refine/demo/demo_image/normal \
    --out infer/Refiner-Sampler4-new-4 \
    --checkpoint Refiner-mirror-vis_mask_c4 \
    --usermemo Inference --infer
  
python _inference.py \
    --input /source/Datasets/SHHQ-1.0_samples/symm \
    --pmask /source/Datasets/SHHQ-1.0_samples/mask_symm/0 \
    --norm_map /source/Datasets/SHHQ-1.0_samples/normal/0 \
    --out infer/Refiner-Sampler4-new-ff \
    --checkpoint Refiner-mirror-vis_mask_c4 \
    --usermemo Inference --infer
  
## PG 2022 version
python _inference.py \
	--input demo/demo_input/tex_1_angle_10005_symmetry.png \
	--pmask demo/demo_input/tex_1_angle_10005_mask_symm.png \
	--norm_map demo/demo_input/tex_1_angle_10005_norm.png \
 	--out infer \
 	--checkpoint rTG-GC-mirror-vis_mask_c \
 	--usermemo Inference --infer
"""