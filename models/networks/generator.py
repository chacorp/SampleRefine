import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from models.networks.architecture import *
from models.networks.base_network import BaseNetwork
import torchvision.utils as tvu


# pix2pix
class UnetGenerator(BaseNetwork):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.init_weights(init_type='normal', gain=0.02)

    def forward(self, input, mask=None, GT=None):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# pix2pixHD
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, mask=None, GT=None): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)
 
## SamplerNet
class SamplerNet(BaseNetwork):
    def __init__(self, 
            input_nc     = 3,
            output_nc    = 2,
            ngf          = 32,
            norm_layer   = nn.InstanceNorm2d,
            padding_type = 'reflect',
            visualize    = False,
            pretrained   = False,
            finetune     = False,
            scale        = 1.0,
            use_gate     = False,
            opt          = None,
        ):
        super().__init__()
        self.visualize = visualize
        self.finetune = finetune
        self.scale = scale

        # normalization
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.opt = opt
        input_nc = 4 if 'vis_mask' in self.opt.concat else 3

        # Apperance encoder
        self.enc_conv0 = Conv2dBlock(input_nc, ngf, 7, 1, 3, bias=use_bias, padding_type=padding_type, norm_layer=norm_layer, use_gate=use_gate, activation='lrelu')
        self.enc_conv1 = ResnetBlock2(ngf,    ngf*2,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.enc_conv2 = ResnetBlock2(ngf*2,  ngf*4,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.enc_conv3 = ResnetBlock2(ngf*4,  ngf*8,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.enc_conv4 = ResnetBlock2(ngf*8,  ngf*16, padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.enc_conv5 = ResnetBlock2(ngf*16, ngf*32, padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')

        # Geometry encoder
        self.norm_conv0 = Conv2dBlock(3, ngf, 7, 1, 3, bias=use_bias, padding_type=padding_type, norm_layer=norm_layer, use_gate=use_gate, activation='lrelu')
        self.norm_conv1 = ResnetBlock2(ngf,    ngf*2,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.norm_conv2 = ResnetBlock2(ngf*2,  ngf*4,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.norm_conv3 = ResnetBlock2(ngf*4,  ngf*8,  padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.norm_conv4 = ResnetBlock2(ngf*8,  ngf*16, padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.norm_conv5 = ResnetBlock2(ngf*16, ngf*32, padding_type=padding_type, norm_layer=norm_layer, downsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')

        # Decoder
        self.dec_conv0 = ResnetBlock2(ngf*64, ngf*16, padding_type=padding_type, norm_layer=norm_layer, upsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.dec_conv1 = ResnetBlock2(ngf*32, ngf*8,  padding_type=padding_type, norm_layer=norm_layer, upsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.dec_conv2 = ResnetBlock2(ngf*16, ngf*4,  padding_type=padding_type, norm_layer=norm_layer, upsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.dec_conv3 = ResnetBlock2(ngf*8,  ngf*2,  padding_type=padding_type, norm_layer=norm_layer, upsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
        self.dec_conv4 = ResnetBlock2(ngf*2,  ngf,    padding_type=padding_type, norm_layer=norm_layer, upsample=True, bias=use_bias, use_gate=use_gate, activation='lrelu')
            
        # sampling grid
        self.toGrid = Conv2dBlock(ngf, 2, 3, 1, 1, bias=use_bias, padding_type=padding_type, norm_layer=None, activation='tanh')

        SIZE = self.opt.data_size
        c = torch.zeros(1, SIZE, SIZE, 2)
        c[..., 0] = torch.linspace(-1, 1, SIZE)
        c[..., 1] = torch.linspace(-1, 1, SIZE).unsqueeze(-1)
        self.grid = c

        if pretrained:
            # path = "output/FTG-GC-mirror-L1-1-lpips-1-step-1-2-3-1.5-boolean_mask-vis_mask/netG/G__iter{:>06}.pth".format(30000)
            # path = "output/Sampler4-mirror-norm_map_vis_mask-L1_10/netG/G__iter{:>06}.pth".format(30000)
            # path = "output/Sampler4-mirror-norm_map_vis_mask/netG/G__iter{:>06}.pth".format(30000)
            path = "output/S1-mirror-norm_map_vis_mask-D/netG/G__iter{:>06}.pth".format(30000)
            
            self.load_state_dict(torch.load(path))

            if self.finetune:
                for param in self.parameters():
                    param.requires_grad = False
            print(f'loaded model successfully: {path}')
            
        else:
            self.init_weights(init_type='normal', gain=0.02)

    def forward(self, x, mask=None, GT=False, mode='s'):
        if not mode in ['g', 's', 's_m', 's_f', 'all']:
            raise NotImplementedError(f'mode not implemented: {mode}')
        with torch.no_grad():
            mesh_grid = self.grid.repeat(x.size(0), 1, 1, 1).to(x)
            mesh_grid_permute = mesh_grid.permute(0, 3, 1, 2)

        if 'vis_mask' in self.opt.concat:
            src_ = torch.cat((x[:,:3], x[:,-2:-1]), dim=1)
        else:
            src_ = x[:, :3]

        norm = x[:, 3:-1]

        enc0 = self.enc_conv0(src_) # 256
        enc1 = self.enc_conv1(enc0) # 128
        enc2 = self.enc_conv2(enc1) # 64
        enc3 = self.enc_conv3(enc2) # 32
        enc4 = self.enc_conv4(enc3) # 16
        enc5 = self.enc_conv5(enc4) # 8

        norm_enc0 = self.norm_conv0(norm) # 256
        norm_enc1 = self.norm_conv1(norm_enc0) # 128
        norm_enc2 = self.norm_conv2(norm_enc1) # 64
        norm_enc3 = self.norm_conv3(norm_enc2) # 32
        norm_enc4 = self.norm_conv4(norm_enc3) # 16
        norm_enc5 = self.norm_conv5(norm_enc4) # 8

        dec0 = self.dec_conv0(torch.cat((norm_enc5, enc5), dim=1)) # 16
        dec1 = self.dec_conv1(torch.cat((dec0, enc4), dim=1)) # 32
        dec2 = self.dec_conv2(torch.cat((dec1, enc3), dim=1)) # 64       
        dec3 = self.dec_conv3(torch.cat((dec2, enc2), dim=1)) # 128
        dec4 = self.dec_conv4(dec3) # 256

        ## add offset to the mesh_grid
        grid_permute = mesh_grid_permute + self.toGrid(dec4) * self.scale         
        grid = grid_permute.permute(0,2,3,1)                    # [B, H, W, 2]

        out = F.grid_sample(x[:,:3], grid, align_corners=True)
        return out

## RefinerNet
class RefinerNet(BaseNetwork):
    def __init__(self, 
            input_nc     = 3,
            output_nc    = 3,
            ngf          = 32,
            norm_layer   = nn.InstanceNorm2d,
            num_layers   = 3,
            num_blocks   = 8,
            use_gate     = True,
            use_dilate   = False,
            padding_type = 'reflect',
            MODE         = 'blend',
            opt          = None,
        ):
        super().__init__()

        # use bias if not BatchNorm2d
        use_bias        = True
        self.data_size  = 256
        self.MODE       = MODE

        if opt != None:
            self.opt        = opt
            num_layers      = opt.num_layers
            num_blocks      = opt.num_blocks
            use_gate        = opt.use_gate
            use_dilate      = opt.use_dilate
            ngf             = opt.ngf
            self.no_skip    = opt.no_skip
            self.MODE       = opt.Refine_mode

        ### texture encoder-decoder:
        self.tEncoder = Encoder(input_nc, ngf, num_layers, norm_layer, padding_type=padding_type, getIntermFeat=True, opt=opt)
        nf = self.tEncoder.output_nc

        # residual blocks
        res_block = []

        d = 1
        self.output_gate = False
        for i in range(num_blocks):
            res_block += [ResnetBlock2(nf, nf, padding_type=padding_type, norm_layer=norm_layer, dilate=d, bias=use_bias, use_gate=use_gate)]

        ### residual blocks
        self.enc_rb = nn.Sequential(*res_block)
        C = 1 if self.no_skip else 2

        ### decoders
        if self.opt.Refine_mode != 'noblend':
            output_nc = output_nc + 1

        self.tDecoder = Decoder(nf*C, output_nc, ngf, num_layers, norm_layer=norm_layer, padding_type=padding_type, opt=opt)

        self.init_weights(init_type='normal', gain=0.02)
        
        if self.opt.mode == 'test' and not self.opt.infer:
            self.num = 0
            os.makedirs(self.opt.out + '/refine', exist_ok=True)
            os.makedirs(self.opt.out + '/mask', exist_ok=True)

    def forward(self, x, mask=None, GT=False, enc_only=False):

        tf = self.tEncoder(x)   # list(tensor)
        r4 = self.enc_rb(tf[-1])# + tf[-1]

        if enc_only:
            return tf
            
        out = self.tDecoder(r4, tf)#, mask=mask)

        if self.opt.Refine_mode != 'noblend':
            out, mask = out[:, :-1], out[:, -1].unsqueeze(1)

        if self.opt.mode == 'test' and not self.opt.infer:
            for b in range(out.size(0)):
                tvu.save_image(out[b],  f"{self.opt.out}/refine/out_{str(self.num).zfill(3)}.png",nrow=1, padding=0)
                if self.opt.Refine_mode != 'noblend':
                    tvu.save_image(mask[b], f"{self.opt.out}/mask/mask_{str(self.num).zfill(3)}.png", nrow=1, padding=0)
                self.num += 1
        return out, mask
