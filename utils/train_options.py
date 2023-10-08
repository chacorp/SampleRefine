from builtins import ValueError
import sys
import argparse
import os
import torch
# from utils.constants import CBLUE, CGREEN, CEND

class TrainOptions():
    def __init__(self):
        self.initialized = False
        self.message = ''
    
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------- End -----------------'
        self.message = message

    def parse(self):
        parser = argparse.ArgumentParser('Generating 3D Human Texture')
        parser.add_argument('--mode', '-m', type=str, default='train', choices=['debug', 'train', 'test', 'infer', 'eval'], help='mode for network')
        parser.add_argument('--port', '-p', type=str, default='8888',  help='port number for tensorboard')

        # setting
        # parser.add_argument('--device_id', nargs='+', type=int, default=[0], help='type in gpu id to use')
        parser.add_argument('--device_id', nargs='+', type=int, default=[i for i in range(torch.cuda.device_count())], help='type in gpu id to use')
        parser.add_argument('--max_workers', '-j',    type=int, default=0, help='max number of workers')
        
        # infer
        parser.add_argument('--infer',          action='store_true',       help='inference mode')
        parser.add_argument('--norm_map',       type=str,default='../Textures', help='normal map')
        parser.add_argument('--pmask',          type=str,default='../Textures', help='partial mask')
        parser.add_argument('--input', '-i',    type=str, default='none',  help='input directory')        
        
        # directory
        parser.add_argument('--data', '-d',     type=str,default='../Textures', help='data set directory')
        parser.add_argument('--out',  '-o',     type=str,                  help='output directory')        
        parser.add_argument('--no_resize',      dest='resize', action='store_false', help='if specified, do *not* resize data image')
        parser.add_argument('--data_size',      type=int, default=256,     help='target size of the data')
        parser.add_argument('--checkpoint',     type=str,                  help='checkpoint folder')
        parser.add_argument('--which_epoch',    type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.set_defaults(resize=True)

        ###[ Optimizer ]###################################################
        parser.add_argument('--lr',             type=float, default=2e-4,  help='learning rate')
        parser.add_argument('--beta1',          type=float, default=0.9,   help='beta 1')
        parser.add_argument('--beta2',          type=float, default=0.999, help='beta 2')

        ###[ Train ]#######################################################
        parser.add_argument('--noTrain',        dest='isTrain',            action='store_false', help='not for Training')
        parser.set_defaults(isTrain=True)
        parser.add_argument('--continue_train', action='store_true',       help='continue Training')

        parser.add_argument('--epochs', '-e',   type=int, default=100,     help='epoch -> it is used as iteration')
        parser.add_argument('--batch_size','-b',type=int, default=8,       help='# of batch size')
        parser.add_argument('--niter',          type=int, default=50,      help='# of iter at starting learning rate. This is NOT the total #epochs. Total #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay',    type=int, default=50,      help='# of iter to linearly decay learning rate to zero')

        parser.add_argument('--gan_mode',       type=str, default='original', help='(ls | original | hinge | w)', choices=['ls', 'original', 'hinge', 'w'])
        parser.add_argument('--gp_weight',      type=float, default=10,    help= 'weight for the w-gan gradient panalty')

        parser.add_argument('--progressive',    action='store_true',       help='apply progressive transformation')
        parser.add_argument('--pg_weight',      type=float, default=0.025, help='weight for the progressive transformation')

        ###[ Validation ]##################################################
        parser.add_argument('--valid_iter',     type=int, default=1000,    help='view angle for the test')

        ###[ Test ]########################################################
        parser.add_argument('--test_angle',     type=int, default=0,       help='view angle for the test')

        ###[ Dataloader ]################################################## 
        ### dataset
        parser.add_argument('--opposite',       action='store_true',       help='add opposite view of the data (STN)')
        parser.add_argument('--triplet_dataset',action='store_true',       help='use triplet dataset')
        parser.add_argument('--mirror',         action='store_true',       help='use mirrored data')
        parser.add_argument('--humbi',          action='store_true',       help='use humbi data for test')
        parser.add_argument('--masking' ,       action='store_true',       help='use masked GT as a input data')

        ## augmentation
        parser.add_argument('--p',              type=float, default=0.8,   help='probability for applying transform')
        parser.add_argument('--translate',      action='store_true',       help='apply translate transform to data')
        parser.add_argument('--sheer',          action='store_true',       help='apply sheer transform to data')
        parser.add_argument('--rotate',         action='store_true',       help='apply rotate transform to data')
        parser.add_argument('--affine',         action='store_true',       help='apply affine transform to data')
        parser.add_argument('--naive_tps',      action='store_true',       help='apply tps transform to data')
        parser.add_argument('--tps',            action='store_true',       help='apply tps transform to data')
        parser.add_argument('--tps_scale',      type=float, default=0.1,   help='scale for applying transform')
        parser.add_argument('--aug_color',      action='store_true',       help='apply color augmentation')
        parser.add_argument('--interval',       type=int,   default=10,    help='interval for azimoth (Euler angle)')

        ### additional data
        parser.add_argument('--concat',         type=str, default='none',  help='concatenate specified to the input',
            choices=['none','vis_mask', 'vis_mask_c', 'norm_map', 'norm_map_vis_mask', 'disp_map_vis_mask', 'norm_map_vis_mask_c'])
        parser.add_argument('--use_consist',    action='store_true',       help='loss for consisting information from partial to generated')
        parser.add_argument('--rendered_data',  action='store_true',       help='if specified, use rendered data as an input')

        ###[ Model ]#######################################################
        parser.add_argument('--Baseline',       action='store_true',       help='if specified, use model from 360 paper')
        parser.add_argument('--SamplerNet',     type=str, default='none',  help='if specified, use sampled texture by SamplerNet as an input',
            choices=['none','vis_mask', 'vis_mask_c', 'norm_map', 'norm_map_vis_mask', 'norm_map_vis_mask_c'])
        parser.add_argument('--FTG_scale',      type=float, default=1,     help='mode for FTG')
        
        ### generator
        parser.add_argument('--G',              type=str, default='TG',    help='generator model', 
            choices=[ 'pix2pix','pix2pixHD','UVGAN','360Degree','S1','S2','R1' ])
        parser.add_argument('--Refine_mode',       type=str, default='blend', help='mode for RefinerNet', 
            choices=[ 'blend', 'add', 'noblend' ])

        parser.add_argument('--num_blocks',     type=int, default=8,       help='# residual blocks')
        parser.add_argument('--num_layers',     type=int, default=3,       help='# down sample layers')
        parser.add_argument('--style',          action='store_true',       help='if specified, encode style and modulate features with Adain')
        parser.add_argument('--ngf',            type=int, default=32,      help='# filters in each gnerator')

        parser.set_defaults(use_mask=True)
        parser.add_argument('--no_use_mask',    dest='use_mask',           action='store_false', help='use un-masked partial map')
        parser.add_argument('--use_gate',       action='store_true',       help='if specified, use Gated Convolution')
        parser.add_argument('--use_dilate',     action='store_true',       help='if specified, use Dilated Convolution')
        parser.add_argument('--no_skip',        action='store_true',       help='if specified, do *not* use skip connection')
        
        ### discriminator
        parser.add_argument('--weight_norm',    type=str, default='none',  help='(sn | wn | none)', choices=['sn', 'wn', 'none'])
        parser.add_argument('--D',              type=str, default='PatchD',help='discriminator model', 
            choices=['PatchD', 'MS-D'])

        parser.add_argument('--dpD',            action='store_true',       help='if specified, add Discriminator for densepose')
        parser.add_argument('--conditional',    action='store_true',       help='if specified, feed data with partial concatenated Discriminator')
        parser.add_argument('--num_D',          type=int, default=1,       help='# discriminators to be used in multiscale')
        parser.add_argument('--ndf',            type=int, default=64,      help='# filters in each discriminator')
        parser.add_argument('--n_layers_D',     type=int, default=4,       help='# layers in each discriminator')
        
        ## adaptive discriminator augmenation
        parser.add_argument("--augment_p",      type=float, default=0.8,   help="probability of applying augmentation. 1 = use adaptive augmentation")
        parser.add_argument("--ada_target",     type=float, default=0.6,   help="target augmentation probability for adaptive augmentation")
        parser.add_argument("--ada_length",     type=int,   default=1000,  help="steps to reach augmentation probability for adaptive augmentation")
                

        ###[ Loss Functions ]##############################################
        ### lamda
        parser.add_argument('--lambda_L1',      type=float, default=1.0,   help='weight for L1 loss')
        parser.add_argument('--lambda_vgg',     type=float, default=1.0,  help='weight for vgg loss')
        parser.add_argument('--lambda_feat',    type=float, default=1.0,  help='weight for feature matching loss')
        parser.add_argument('--lambda_LPIPS',   type=float, default=1.0,   help='weight for LPIPS loss')
        parser.add_argument('--lambda_gan',     type=float, default=1.0,   help='weight for gan loss')

        parser.add_argument('--lambda_render',  type=float, default=0.0,   help='weight for render loss')

        ### for rendering
        parser.add_argument('--render_size',    type=int,  default=512,    help='size of rendered image.')
        parser.add_argument('--render_distance',type=float,default=1.0,    help='distance for rendering camera.')
        parser.add_argument('--render_scale',   type=float,default=1.0,    help='scale for rendering camera.')
        parser.add_argument('--render_at',      nargs='+', type=float,     default=[0.0, 0.9, 0.0], help='position for rendering camera')
        parser.add_argument('--render_elev',    nargs='+', type=float,     default=[0.0], help='elevation for rendering camera')
        parser.add_argument('--render_azim',    nargs='+', type=float,     default=[0.0], help='azimuth for rendering camera')

        ### for tensor board
        parser.add_argument('--usermemo',       type=str,                  help='memo which will be added to the checkpoint file')
        opt = parser.parse_args()

        if opt.data_size != 512:
            opt.resize = True

        opt.render_at   = tuple(opt.render_at,)
        opt.render_elev = tuple(opt.render_elev,)
        opt.render_azim = tuple(opt.render_azim,)
            
        if opt.progressive: # default == tps
            if not opt.rotate and not opt.translate and not opt.tps and not opt.affine and not opt.naive_tps and not opt.sheer:
                opt.tps         = True
            opt.tps_scale       = 0.0  # no TPS from start
            opt.p               = 0.8  # prob applying transform

        self.parser             = parser
        self.print_options(opt)

        self.initialized        = True
        return opt