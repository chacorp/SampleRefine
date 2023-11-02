from builtins import ValueError
import os
from os.path import join
from glob import glob

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import optim

import torch.nn.functional as F

###[ model ]##################################################
from models.network import (
    UNetGenerator360,
    PatchDiscriminator360,
)
from models.networks.generator import *
from models.networks.discriminator import *

import lpips
from piq import ssim
from kornia.losses.psnr import PSNRLoss

from utils.loss import GANLoss, VGGLoss
from utils.dataloader import (
    _get_mask, 
    _get_part_mask_bn, 
    augment_blur
)
from utils.transformation import (
    tps_transform_kornia, 
    affine_transform, 
    rotate_transform, 
    sheer_transform, 
    translate_transform, 
    naive_tps_transform_kornia
)
from utils.constants import BOX

class Model(nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.device = device
        self.opt = opt 
        self.continue_epoch = 0

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor

        # networks (netD will be ignored when training SamplerNet)
        self.netG, self.netD = self.initialize_networks()
        # SamplerNet for RefinerNet
        if self.opt.SamplerNet != 'none':
            self.sampler = self.get_sampler_net(KEY='S1')

        # if opt.mode in ['train', 'debug'] :
        #     self.optimizer_G, self.optimizer_D = self.create_optimizers()

        ### Loss
        self.criterionL1     = nn.L1Loss()
        self.criterionGAN    = GANLoss(gan_mode=opt.gan_mode, tensor=self.FloatTensor)
        self.criterionVGG    = VGGLoss(device=self.device)

        if opt.mode !='test':
            if self.opt.lambda_render > 0:
                from utils.renderer import Renderer
                self.extend   = len(self.opt.render_azim) * len(self.opt.render_elev)
                if self.extend < 1:
                    raise ValueError(f'self.extend needs to be grater than 1: azim={self.opt.render_azim}, elev={self.opt.render_elev}')
                self.renderer = Renderer(device=device, extend=self.extend, opt=opt)

        ### lpips
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_fn.eval()
        
        ### psnr
        self.psnr = PSNRLoss(max_val=1.0)

        self.box            = torch.ceil(BOX * self.opt.data_size / 512).type(torch.int32)
        self.label_mask_bn  = _get_part_mask_bn(resize=self.opt.resize, size=self.opt.data_size).to(self.device)
        self.label_mask     = self.label_mask_bn * 1.0
        self.uv_mask        = _get_mask(resize=self.opt.resize, size=self.opt.data_size, bn=False).to(self.device)
    
        ### get transform augmentation
        self.init_augmentation()
                
        # self.old_lr = opt.lr

    def init_augmentation(self):        
        if self.opt.progressive:
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
            self.tps_scale = self.opt.tps_scale
        else:
            self.tps_scale = 0
        
    def use_gpu(self):
        return len(self.opt.device_id) > 0

    def get_sampler_net(self, KEY):        
        inputG_nc = 3
        if 'vis_mask' in self.opt.SamplerNet:
            inputG_nc = inputG_nc + 1
        if 'norm_map' in self.opt.SamplerNet:
            inputG_nc = inputG_nc + 3
        
        # ---------------------------------------------------------- EG 2023 submission
        if KEY == 'S1':
            netG = SamplerNet(inputG_nc, use_gate=False, visualize=False, pretrained=True, opt=self.opt)
        # -----------------------------------------------------------------------------
        elif KEY == 'S2':
            netG = SamplerNet2(inputG_nc, use_gate=True, visualize=False, pretrained=True,  opt=self.opt)
        else:
            raise ValueError(f"moded not implemented!: {KEY}")
        self.sampler = netG.to(self.device)
        self.sampler.eval()
        return self.sampler

    def initialize_networks(self):
        inputG_nc  = 3
        inputD_nc  = 3
        outputG_nc = 3

        if self.opt.Baseline:
            # concatenates gaussian noise in the model
            netG = UNetGenerator360(inputG_nc+3, outputG_nc)

            # concatenates input(partial map) and output(full texture map)
            netD = PatchDiscriminator360(inputD_nc+3)

            if self.opt.mode == 'test':
                netG = self.load_model(netG, model='G', checkpoint=self.opt.checkpoint)
            elif self.opt.continue_train:
                netG = self.load_model(netG, model='G', checkpoint=self.opt.checkpoint)
                netD = self.load_model(netD, model='D', checkpoint=self.opt.checkpoint)

            return netG.to(self.device), netD.to(self.device)

        if self.opt.conditional:
            inputD_nc += 3 # concat partial and full

        if 'vis_mask' in self.opt.concat:
            inputG_nc += 1
        if 'norm_map' in self.opt.concat:
            inputG_nc += 3
        
        #### for the evaluation w/ Albahar et al.
        if self.opt.opposite:
            inputG_nc += 3
            inputD_nc += 3

        # generator
        if self.opt.G == '360Degree':
            netG = UNetGenerator360(6, outputG_nc)
        elif self.opt.G == 'pix2pix':
            netG = UnetGenerator(inputG_nc, outputG_nc, num_downs=3, norm_layer=nn.InstanceNorm2d)
        elif self.opt.G == 'pix2pixHD':
            netG = LocalEnhancer(inputG_nc, outputG_nc, norm_layer=nn.InstanceNorm2d)
            
        # ---------------------------------------------------------- EG 2023 submission
        elif self.opt.G == 'S1': 
            netG = SamplerNet(inputG_nc, outputG_nc, use_gate=False, opt=self.opt)
        # -----------------------------------------------------------------------------
        
        # elif self.opt.G == 'S2':
        #     netG = SamplerNet2(inputG_nc, outputG_nc, use_gate=True, condition=self.opt.concat,  opt=self.opt)
        elif self.opt.G == 'R1':
            netG = RefinerNet(inputG_nc, outputG_nc, opt=self.opt)
        else:
            raise ValueError(f"moded not implemented!: {self.opt.G}")
            
        # discriminator
        if self.opt.G == '360Degree':
            netD = PatchDiscriminator360(6)
        else:
            if self.opt.D == 'MS-D':
                netD = MultiscaleDiscriminator(inputD_nc, opt=self.opt)
            elif self.opt.D == 'PatchD':
                netD = PatchDiscriminator(inputD_nc, opt=self.opt)
            else:
                raise ValueError(f"moded not implemented!: {self.opt.D}")

        if self.opt.mode == 'test':
            netG = self.load_model(netG, model='G', checkpoint=self.opt.checkpoint)
        elif self.opt.continue_train:
            netG = self.load_model(netG, model='G', checkpoint=self.opt.checkpoint)
            netD = self.load_model(netD, model='D', checkpoint=self.opt.checkpoint)

        return netG.to(self.device), netD.to(self.device)

    def load_model(self, net, model, checkpoint, epoch=None):
        if self.opt.which_epoch == 'latest':
            path = sorted(glob(f"output/{checkpoint}/net{model}/*"))[-1]
            if 'epoch' in path:
                e = path.split('epoch')[-1].split('.')[0]
                self.continue_epoch = int(e)
            elif 'iter' in path:
                e = path.split('iter')[-1].split('.')[0]
                self.continue_epoch = int(e)
            else:
                raise ValueError('invalid file in path!: {}'.format(path))
        else:
            files = sorted(glob(f"output/{checkpoint}/net{model}/*"))
            if 'epoch' in files[0]:
                keyword = 'epoch'
            else:
                keyword = 'iter'
            indices = [file.split(keyword)[-1].split('.')[0] for file in files]
            for i, idx in enumerate(indices):
                if self.opt.which_epoch in idx: 
                    self.continue_epoch = int(idx)
                    indices = i 
                    break
            assert type(indices) == int, 'invalid number of epoch! it must be multiple of 10'

            path = files[indices]
        net.load_state_dict(torch.load(path))
        print(f'loaded model successfully: {path}')
        return net

    @torch.no_grad()
    def preprocess_data(self, data):
        """
        TODO:
        data.keys() will be modified from:
        
        {'input', 'target', 'pmask', 'norm_map'}
            -> {'T_inputA', 'T_inputB', 'Vis_maskA', 'Vis_maskB', 'GT_texture', 'norm_map', 'has_GT'}
            
        change below codes according to the new attribute
        """
        d = {}
        d['real_image'] = data['GT_texture'].to(self.device)
            
        d['vis_maskA']   = data['Vis_maskA'].to(self.device)
        d['vis_maskB']   = data['Vis_maskB'].to(self.device)
        
        if self.opt.rendered_data:
            d['rendered'] = data['rendered'].to(self.device)

        d['part_imageA'] = data['T_inputA'].to(self.device)
        d['part_imageB'] = data['T_inputB'].to(self.device)
            
        # for progressive augmentation
        if self.opt.masking:
            d['part_imageA'] = data['maskingA'].to(self.device)
            d['part_imageB'] = data['maskingB'].to(self.device)
            d['denseposeA']  = data['T_inputA'].to(self.device)
            d['denseposeB']  = data['T_inputB'].to(self.device)
        
        ## re-normalize 0 ~ 1 >>> -1 ~ 1
        if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
            d['real_image'] = (d['real_image'] * 2) -1
            d['part_imageA'] = (d['part_imageA'] * 2) -1
            d['part_imageB'] = (d['part_imageB'] * 2) -1
            if self.opt.progressive and self.opt.masking:
                d['denseposeA'] = (d['denseposeA'] * 2) -1
                d['denseposeB'] = (d['denseposeB'] * 2) -1

        if self.opt.SamplerNet != 'none':
            with torch.no_grad():
                # imageA
                s_inputA = torch.cat((
                        d['part_imageA'], 
                        data['norm_map'].to(self.device), 
                        d['vis_maskA']
                    ), dim=1)
                sampled_texture = self.sampler(s_inputA, mode='s')
                d['sampled_imageA'] = sampled_texture.detach().requires_grad_()   # [B, 3, H, W]
                # imageB
                s_inputB = torch.cat((
                        d['part_imageB'], 
                        data['norm_map'].to(self.device), 
                        d['vis_maskB']
                    ), dim=1)
                sampled_texture = self.sampler(s_inputB, mode='s')
                d['sampled_imageB'] = sampled_texture.detach().requires_grad_()   # [B, 3, H, W]

        if self.opt.G in ['R1', 'R2',]:
            d['part_imageA'] = d['sampled_imageA']
            d['part_imageB'] = d['sampled_imageB']

        if self.opt.progressive:
            d = self.apply_transform(d)
        
        if 'norm_map' in self.opt.concat:
            concat = self.opt.concat[:8] # removing _vis_mask
            d[concat] = data[concat].to(self.device)
            
        
        return d

    @torch.no_grad()
    def apply_blur(self, data):
        for key in ['part_image', 'densepose', 'sampled_image']:
            if key in data.keys():
                data[key] = augment_blur(data[key], self.opt.p)
                # data[key] = augment_blur(data[key], 0.5)
        return data

    @torch.no_grad()
    def apply_transform(self, data):
        # tps_scale updated in trainer code!
        if self.tps_scale > 0:
            if self.tps_scale > 0.125 and torch.rand(1) > 0.5: 
                data['part_imageA'] = data['denseposeA'] if self.opt.masking else data['part_imageA']
                data['part_imageB'] = data['denseposeB'] if self.opt.masking else data['part_imageB']
            else: 
                # for imageA
                C_A = data['part_imageA'].size(1)
                tempA = torch.cat((data['part_imageA'], data['vis_maskA']), dim=1)
                tempA = self.transform(
                        img         = tempA,
                        p           = self.opt.p,
                        scale       = self.tps_scale,
                        input_size  = self.opt.data_size,
                        label_mask  = self.label_mask,
                        device      = self.device
                    )
                data['part_imageA'], data['vis_maskA'] = tempA[:, :C_A], tempA[:, C_A:]
                # for imageB
                C_B = data['part_imageB'].size(1)
                tempB = torch.cat((data['part_imageB'], data['vis_maskB']), dim=1)
                tempB = self.transform(
                        img         = tempB,
                        p           = self.opt.p,
                        scale       = self.tps_scale,
                        input_size  = self.opt.data_size,
                        label_mask  = self.label_mask,
                        device      = self.device
                    )
                data['part_imageB'], data['vis_maskB'] = tempA[:, :C_B], tempB[:, C_B:]
        return data

    @torch.no_grad()
    def inference(self, datas):
        if self.opt.G in ['R1','R2']:
            (fake_refineA, maskA), (fake_refineB, maskB) = self.generate_fake(datas, is_train=False)
            if self.opt.SamplerNet != 'none':
                part_imageA = datas['sampled_imageA']
                part_imageB = datas['sampled_imageB'] 
            else:
                part_imageA = datas['part_imageA']
                part_imageB = datas['part_imageB']
                
            if self.opt.Refine_mode == 'blend':
                fake_imageA = part_imageA * maskA + fake_refineA * (1-maskA)
                fake_imageB = part_imageB * maskB + fake_refineB * (1-maskB)
            elif self.opt.Refine_mode == 'add':
                fake_imageA = part_imageA + fake_refineA * (1-maskA)
                fake_imageB = part_imageB + fake_refineB * (1-maskB)
            elif self.opt.Refine_mode == 'residual':
                fake_imageA = part_imageA + fake_refineA
                fake_imageB = part_imageB + fake_refineB
            elif self.opt.Refine_mode == 'noblend':
                fake_imageA = fake_refineA
                fake_imageB = fake_refineB
        else:
            fake_imageA, fake_imageB = self.generate_fake(datas, is_train=False)
        return fake_imageA, fake_imageB

    def forward(self, data, mode):
        datas = self.preprocess_data(data)

        if mode == 'generator':
            g_loss = self.compute_generator_loss(datas)
            return g_loss

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(datas)
            return d_loss

        elif 'visualize' in mode:
            with torch.no_grad():
                if self.opt.progressive and self.opt.masking:
                    if 'valid' in mode:
                        datas['part_imageA'] = datas['denseposeA']
                        datas['part_imageB'] = datas['denseposeB']
                if self.opt.SamplerNet != 'none':
                    datas['part_imageA'] = datas['sampled_imageA']
                    datas['part_imageB'] = datas['sampled_imageB']

                fake_imageA, fake_imageB = self.inference(datas)

                if self.opt.SamplerNet != 'none':
                    part_imageA = datas['sampled_imageA']
                    part_imageB = datas['sampled_imageB']
                elif self.opt.rendered_data:
                    part_imageA = datas['rendered']
                    part_imageB = datas['rendered']
                else:
                    part_imageA = datas['part_imageA']
                    part_imageB = datas['part_imageB']

                if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
                    fake_imageA = (fake_imageA + 1) * 0.5
                    part_imageA = (part_imageA + 1) * 0.5
                    fake_imageB = (fake_imageB + 1) * 0.5
                    part_imageB = (part_imageB + 1) * 0.5
                return part_imageA, fake_imageA, part_imageB, fake_imageB
                
        elif mode == 'validation':
            with torch.no_grad():
                g_loss, fake_imageA, fake_imageB = self.compute_generator_loss(datas, is_train=False)
                
                if self.opt.G not in ['S1', 'S2']:
                    d_loss = self.compute_discriminator_loss(datas, is_train=False)
                else:
                    d_loss = {'no D loss':torch.zeros([1]).to(self.device)}
                
                fake_image_maskedA = fake_imageA*self.uv_mask
                fake_image_maskedB = fake_imageB*self.uv_mask
                real_image = datas['real_image']*self.uv_mask

                _lpipsA, _psnrA, _ssimA = self.compute_evaluate(fake_image_maskedA, real_image)
                _lpipsB, _psnrB, _ssimB = self.compute_evaluate(fake_image_maskedB, real_image)

                if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
                    fake_imageA = (fake_imageA + 1) * 0.5
                    fake_imageB = (fake_imageB + 1) * 0.5
                return g_loss, d_loss, _lpipsA, _psnrA, _ssimA, _lpipsB, _psnrB, _ssimB 

        elif mode == 'test':
            with torch.no_grad():
                if self.opt.progressive and self.opt.masking:                
                    datas['part_imageA'] = datas['denseposeA']               
                    datas['part_imageB'] = datas['denseposeB']
                if self.opt.SamplerNet != 'none':
                    datas['part_imageA'] = datas['sampled_imageA']
                    datas['part_imageB'] = datas['sampled_imageB']

                fake_imageA, fake_imageB = self.inference(datas)

                if self.opt.rendered_data:
                    part_imageA = datas['rendered']
                    part_imageB = datas['rendered']
                else:
                    part_imageA = datas['part_imageA']
                    part_imageB = datas['part_imageB']

                fake_image_maskedA = fake_imageA*self.uv_mask
                fake_image_maskedB = fake_imageB*self.uv_mask
                real_image = datas['real_image']*self.uv_mask

                _lpipsA, _psnrA, _ssimA = self.compute_evaluate(fake_image_maskedA, real_image)
                _lpipsB, _psnrB, _ssimB = self.compute_evaluate(fake_image_maskedB, real_image)
                if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
                    fake_imageA = (fake_imageA + 1) * 0.5
                    part_imageA = (part_imageA + 1) * 0.5
                    fake_imageB = (fake_imageB + 1) * 0.5
                    part_imageB = (part_imageB + 1) * 0.5
                return {**_lpipsA, **_psnrA, **_ssimA},{**_lpipsB, **_psnrB, **_ssimB}, part_imageA, fake_imageA, part_imageB, fake_imageB

        else:
            raise ValueError("|mode| is invalid")

    def save(self, path, epoch):
        self.save_model(self.netG, model='G', path=path, epoch=epoch)
        self.save_model(self.netD, model='D', path=path, epoch=epoch)

    def save_model(self, net, model, path, epoch=0):
        # make dir
        save_Path = f"{path}/net{model}"
        os.makedirs(save_Path, exist_ok=True)

        # name file
        # save_Path = join(save_Path,  "{}__epoch{:>03}.pth".format(model, epoch))
        save_Path = join(save_Path,  "{}__iter{:>06}.pth".format(model, epoch))
        torch.save(net.cpu().state_dict(), save_Path)

        if len(self.opt.device_id) and torch.cuda.is_available():
            net.to(self.device)

    def create_optimizers(self):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = self.opt.beta1, self.opt.beta2
        G_lr, D_lr = self.opt.lr, self.opt.lr

        optimizer_G = optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D
    
    def generate_fake(self, datas, is_train=True, GT=False):
        if is_train:
            self.netG.train()
        else: 
            self.netG.eval()

        part_imageA, part_maskA = datas['part_imageA'], datas['vis_maskA']
        part_imageB, part_maskB = datas['part_imageB'], datas['vis_maskB']

        if self.opt.SamplerNet != 'none':
            with torch.no_grad():
                part_imageA = datas['sampled_imageA']
                part_imageB = datas['sampled_imageB']
        ### concat or add
        if self.opt.concat != 'none':
            if self.opt.concat == 'vis_mask_c':                                     # occlusion mask
                part_imageA = torch.cat((part_imageA, (1-part_maskA)), dim=1)    
                part_imageB = torch.cat((part_imageB, (1-part_maskB)), dim=1)       
            elif self.opt.concat == 'vis_mask':                                     # visibility mask
                part_imageA = torch.cat((part_imageA, part_maskA), dim=1)
                part_imageB = torch.cat((part_imageB, part_maskB), dim=1)
            elif self.opt.concat == 'norm_map':                                     # normal map [B,3,H,W]
                part_imageA = torch.cat((part_imageA, datas['norm_map']), dim=1)
                part_imageB = torch.cat((part_imageB, datas['norm_map']), dim=1)
            elif self.opt.concat == 'norm_map_vis_mask':                            # normal map + vis_mask [B,6,H,W]
                part_imageA = torch.cat((part_imageA, datas['norm_map'], part_maskA), dim=1)
                part_imageB = torch.cat((part_imageB, datas['norm_map'], part_maskB), dim=1)
            elif self.opt.concat == 'norm_map_vis_mask_c':                          # normal map + vis_mask [B,6,H,W]
                part_imageA = torch.cat((part_imageA, datas['norm_map'], (1-part_maskA)), dim=1)
                part_imageB = torch.cat((part_imageB, datas['norm_map'], (1-part_maskB)), dim=1)
            else:
                raise NotImplementedError("Wrong value for opt.concat: {}".format(self.opt.concat))

        if self.opt.Baseline or self.opt.G == '360Degree':
            randnA      = torch.randn(*part_imageA.shape).to(self.device)
            part_imageA = torch.cat((part_imageA, randnA), dim=1)
            randnB      = torch.randn(*part_imageB.shape).to(self.device)
            part_imageB = torch.cat((part_imageB, randnB), dim=1)

        if self.opt.rendered_data:
            part_imageA = datas['rendered']
            part_imageB = datas['rendered']

        ### forward path
        outA = self.netG(part_imageA, part_maskA, GT=GT) # fake_refineA, maskA
        outB = self.netG(part_imageB, part_maskB, GT=GT) # fake_refineB, maskB
        return outA, outB
        
    def discriminate(self, part_image, fake_image, real_image, datas, is_train=True):
        if is_train:
            self.netD.train()
        else:
            self.netD.eval()

        # In Batch Normalization, the fake and real images are recommended 
        # to be in the same batch to avoid disparate statistics in fake and real images.
        # So both fake and real images are fed to D all at once.

        if self.opt.conditional:
            if self.opt.concat != 'none':
                cond = self.opt.concat[:8]
                fake = torch.cat([datas[cond], fake_image], dim=1)
                real = torch.cat([datas[cond], real_image], dim=1)
            else:
                fake = torch.cat([part_image, fake_image], dim=1)
                real = torch.cat([part_image, real_image], dim=1)
        else:
            fake = fake_image
            real = real_image

        # when using batch norm
        fake_and_real = torch.cat([fake, real], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                # multi-scale discriminator with intermediate feature
                if type(p) == list:
                    fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                    real.append([tensor[tensor.size(0) // 2:] for tensor in p])
                # single discriminator with intermediate feature
                else:
                    fake.append(p[:p.size(0) // 2])
                    real.append(p[p.size(0) // 2:])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
   
    def compute_generator_loss(self, datas, is_train=True):
        part_imageA = datas['part_imageA']
        part_imageB = datas['part_imageB']
        real_image  = datas['real_image']
        
        G_losses = {}
        
        # RefinerNet
        if self.opt.G in ['R1', 'R2']:
            (fake_refineA, maskA), (fake_refineB, maskB) = self.generate_fake(datas, is_train)
            if self.opt.Refine_mode == 'blend':
                fake_imageA = part_imageA * maskA + fake_refineA * (1-maskA)
                fake_imageB = part_imageB * maskB + fake_refineB * (1-maskB)
            elif self.opt.Refine_mode == 'add':
                fake_imageA = part_imageA + fake_refineA * (1-maskA)
                fake_imageB = part_imageB + fake_refineB * (1-maskB)
            elif self.opt.Refine_mode == 'residual':
                fake_imageA = part_imageA + fake_refineA
                fake_imageB = part_imageB + fake_refineB
            elif self.opt.Refine_mode == 'noblend':
                fake_imageA = fake_refineA
                fake_imageB = fake_refineB            
        # SamplerNet
        else:
            fake_imageA, fake_imageB = self.generate_fake(datas, is_train)
            
        fake_imageA = fake_imageA * self.uv_mask
        fake_imageB = fake_imageB * self.uv_mask

        ### for visualization
        # generatedA = fake_imageA.clone().detach()
        # generatedB = fake_imageB.clone().detach()
        
        # masking valid region (real_image is already masked)
        real_image = real_image * self.uv_mask

        # bool_pmask = (part_mask > 0).all(1).unsqueeze(1).repeat(1,3,1,1)

        pred_fakeA, pred_realA = self.discriminate(part_imageA, fake_imageA, real_image, datas, is_train)
        pred_fakeB, pred_realB = self.discriminate(part_imageB, fake_imageB, real_image, datas, is_train)

        ### Reconstruction loss
        if self.opt.lambda_L1:
            if self.opt.G in ['R1', 'rTG', 'bTG']:
                G_losses['L1_A'] = self.compute_L1(fake_imageA, part_imageA) * self.opt.lambda_L1   
                G_losses['L1_B'] = self.compute_L1(fake_imageB, part_imageB) * self.opt.lambda_L1
            else:
                G_losses['L1_A']  = self.compute_L1(fake_imageA, real_image) * self.opt.lambda_L1
                G_losses['L1_B']  = self.compute_L1(fake_imageB, real_image) * self.opt.lambda_L1
                ## [additional loss to original paper]
                G_losses['L1_AB'] = self.compute_L1(fake_imageA, fake_imageB)
        
        ### perceptual loss (LPIPS)
        if self.opt.lambda_LPIPS > 0:
            G_losses['lpipsA'] = self.lpips_fn(fake_imageA, real_image, normalize=True) * self.opt.lambda_LPIPS
            G_losses['lpipsB'] = self.lpips_fn(fake_imageB, real_image, normalize=True) * self.opt.lambda_LPIPS

        # adverserial loss
        if self.opt.lambda_gan > 0:
            g_fakeA = self.criterionGAN(pred_fakeA, target_is_real=True, for_discriminator=False) * self.opt.lambda_gan
            g_fakeB = self.criterionGAN(pred_fakeB, target_is_real=True, for_discriminator=False) * self.opt.lambda_gan
            G_losses['GAN_A'] = g_fakeA
            G_losses['GAN_B'] = g_fakeB

        # feature matching loss
        if self.opt.lambda_feat > 0:
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            # for each discriminator
            for i in range(self.opt.num_D):
                # exclude final prediction
                num_intermediate_outputs = len(pred_fakeA[i]) - 1
                # for each layer output
                for j in range(num_intermediate_outputs):
                    GAN_Feat_loss += self.criterionL1(pred_fakeA[i][j], pred_realA[i][j].detach()) 
                    GAN_Feat_loss += self.criterionL1(pred_fakeB[i][j], pred_realA[i][j].detach()) 
            G_losses['GAN_Feat'] = GAN_Feat_loss * self.opt.lambda_feat / self.opt.num_D
            
        # vgg loss
        if self.opt.lambda_vgg > 0:
            G_losses['VGG'] = self.criterionVGG(fake_imageA, real_image) * self.opt.lambda_vgg
            G_losses['VGG'] = self.criterionVGG(fake_imageB, real_image) * self.opt.lambda_vgg
        
        # Renderloss
        if self.opt.lambda_render > 0:
            fake_rendered, random_list = self.renderer(fake_imageA, None)
            with torch.no_grad():
                real_rendered, _       = self.renderer(real_image, random_list)
            # boolean mask
            fake_mask = (fake_rendered[:,3] > 0).unsqueeze(1).repeat(1,3,1,1)
            real_mask = (real_rendered[:,3] > 0).unsqueeze(1).repeat(1,3,1,1)
            G_losses['render'] = self.criterionL1(fake_rendered[:,:3][fake_mask], real_rendered[:,:3][real_mask]) * self.opt.lambda_render

        return G_losses

    def compute_discriminator_loss(self, datas, is_train=True):
        part_imageA = datas['part_imageA']
        part_imageB = datas['part_imageB']
        real_image  = datas['real_image']
        
        D_losses = {}
        if self.opt.SamplerNet != 'none':
            part_imageA = datas['sampled_imageA']
            part_imageB = datas['sampled_imageB']

        with torch.no_grad():
            if self.opt.G in ['R1', 'R2']:
                (fake_refineA, maskA), (fake_refineB, maskB) = self.generate_fake(datas, is_train=False)
                if self.opt.Refine_mode == 'blend':
                    fake_imageA = part_imageA * maskA + fake_refineA * (1-maskA)
                    fake_imageB = part_imageB * maskB + fake_refineB * (1-maskB)
                elif self.opt.Refine_mode == 'add':
                    fake_imageA = part_imageA + fake_refineA * (1-maskA)     
                    fake_imageB = part_imageB + fake_refineB * (1-maskB)
                elif self.opt.Refine_mode == 'residual':
                    fake_imageA = part_imageA + fake_refineA
                    fake_imageB = part_imageB + fake_refineB
                elif self.opt.Refine_mode == 'noblend':
                    fake_imageA = fake_refineA
                    fake_imageB = fake_refineB
                    
                fake_refineA = fake_refineA * self.uv_mask
                fake_refineA = fake_refineA.detach().requires_grad_() 
                fake_refineB = fake_refineB * self.uv_mask
                fake_refineB = fake_refineB.detach().requires_grad_() 
                # fake_real = real_image * (1-mask)
                # fake_real = fake_real.detach().requires_grad_()

            else:
                fake_imageA, fake_imageB = self.generate_fake(datas, is_train=False)

            fake_imageA = fake_imageA.detach().requires_grad_() 
            fake_imageA = fake_imageA * self.uv_mask
            real_image = real_image * self.uv_mask
            fake_imageB = fake_imageB.detach().requires_grad_() 
            fake_imageB = fake_imageB * self.uv_mask
        
        pred_fakeA, pred_realA = self.discriminate(part_imageA, fake_imageA, real_image, datas, is_train=is_train)
        pred_fakeB, pred_realB = self.discriminate(part_imageB, fake_imageB, real_image, datas, is_train=is_train)

        d_fakeA = self.criterionGAN(pred_fakeA, target_is_real=False, for_discriminator=True)
        d_realA = self.criterionGAN(pred_realA, target_is_real=True,  for_discriminator=True)
        d_fakeB = self.criterionGAN(pred_fakeB, target_is_real=False, for_discriminator=True)
        d_realB = self.criterionGAN(pred_realB, target_is_real=True,  for_discriminator=True)

        if self.opt.gan_mode == 'w':
            D_losses['D_distanceA'] = d_fakeA + d_realA
            D_losses['D_distanceB'] = d_fakeB + d_realB
        else:
            D_losses['D_FakeA'] = d_fakeA
            D_losses['D_realA'] = d_realA
            D_losses['D_FakeB'] = d_fakeB
            D_losses['D_realB'] = d_realB
        
        if self.opt.gan_mode == 'w' and is_train:
            # cannot calculate gp when no_grad is set, such as validation
            D_losses['D_gp_A'] = self._gradient_penalty(part_imageA, fake_imageA, real_image)
            D_losses['D_gp_B'] = self._gradient_penalty(part_imageB, fake_imageB, real_image)

        return D_losses

    def _gradient_penalty(self, part_image, fake_image, real_image):
        ### implementation reference: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
        gradient_penalty = self.FloatTensor(1).fill_(0)

        batch_size = real_image.shape[0]
        # epsilon = 1e-16

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand(*real_image.shape).to(self.device)
        
        with torch.no_grad():
            interpolated = alpha * real_image + (1 - alpha) * fake_image
        interpolated.requires_grad_() 
        interpolated = torch.cat([part_image, interpolated], dim=1)

        # Calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # if self.opt.num_D > 1:
        if self.opt.lambda_feat > 0: # feature matching
            # Calculate gradients of probabilities with respect to examples
            for i in range(self.opt.num_D):
                # output_prob_mean = prob_interpolated[i][-1].mean([-2,-1])
                output_prob = prob_interpolated[i][-1]

                gradients = autograd.grad(
                    outputs = output_prob, 
                    inputs = interpolated,
                    grad_outputs = torch.ones(output_prob.size()).to(self.device),# if self.use_gpu() else torch.ones(prob_interpolated.size()),
                    create_graph = True,
                    retain_graph = True,
                    only_inputs  = True
                )[0]

                gradients = gradients.reshape(batch_size, -1)
                gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() / self.opt.num_D
        else:
            # prob_interpolated = prob_interpolated.mean([-2,-1], True)
            # Calculate gradients of probabilities with respect to examples
            gradients = autograd.grad(
                outputs = prob_interpolated, 
                inputs = interpolated,
                grad_outputs = torch.ones(prob_interpolated.size()).to(self.device),
                create_graph = True, 
                retain_graph = True, 
                only_inputs  = True
            )[0]

            # Gradients have shape (batch_size, num_channels, img_width, img_height),
            # so flatten to easily take norm per example in batch
            gradients = gradients.reshape(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Return gradient penalty
        # return self.opt.gp_weight * (((gradients + epsilon).norm(2, dim=1) - 1) ** 2).mean()
        return self.opt.gp_weight * gradient_penalty

    def compute_L1(self, fake_image, real_image):
        
        ### RefinerNet
        if self.opt.SamplerNet != 'none':
            L1_loss = self.criterionL1(fake_image, real_image)

        ### SamplerNet
        else:
            face_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[0]], real_image[...,self.label_mask_bn[0]])
            body_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[1]], real_image[...,self.label_mask_bn[1]])
            legs_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[2]], real_image[...,self.label_mask_bn[2]])
            arms_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[3]], real_image[...,self.label_mask_bn[3]])
            foot_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[4]], real_image[...,self.label_mask_bn[4]])
            hand_loss  = self.criterionL1(fake_image[..., self.label_mask_bn[5]], real_image[...,self.label_mask_bn[5]])
            L1_loss    = 6*face_loss+body_loss+legs_loss+arms_loss+foot_loss+hand_loss

        return L1_loss

    @torch.no_grad()
    def compute_lpips(self, fake_image, real_image):
        if not self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or not self.opt.Baseline:
            fake_image = (fake_image * 2) - 1
            real_image = (real_image * 2) - 1
        return { 'lpips' : self.lpips_fn(fake_image, real_image).mean() }

    @torch.no_grad()
    def compute_psnr(self, fake_image, real_image):
        if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
            fake_image = (fake_image + 1) * 0.5
            real_image = (real_image + 1) * 0.5
        # -1 is multiplied for the loss term
        return { 'psnr': self.psnr(fake_image, real_image) * -1.0 }

    @torch.no_grad()
    def compute_ssim(self, fake_image, real_image):
        if self.opt.G in ['pix2pix','pix2pixHD','UVGAN','360Degree'] or self.opt.Baseline:
            fake_image = (fake_image + 1) * 0.5
            real_image = (real_image + 1) * 0.5
        fake_image = torch.clamp(fake_image, 0, 1.0)
        return { 'ssim' : ssim(fake_image, real_image, data_range=1.0) }

    def compute_evaluate(self, fake_image, real_image):
        _lpips = self.compute_lpips(fake_image, real_image)
        _psnr  = self.compute_psnr(fake_image,  real_image)
        _ssim  = self.compute_ssim(fake_image,  real_image)
        return _lpips, _psnr, _ssim