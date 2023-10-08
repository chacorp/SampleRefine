import functools
import torch
import torch.nn as nn

from utils.utils import *


def get_padding(padding_type='zero', padding=0):
    if padding_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif padding_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    elif padding_type == 'zero':
        return nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError("Unexpected padding type: {}".format(padding_type))

def get_weightnorm(weight_norm='none'):
    if weight_norm == 'sn':
        return spectral_norm 
    elif weight_norm == 'wn':
        return nn.utils.weight_norm
    elif weight_norm == 'none':
        return None
    else:
        raise NotImplementedError("Unexpected weight norm!: {}".format(weight_norm))

def get_activation(activation='none'):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'none':
        return None
    else:
        raise NotImplementedError("Unexpected activation: {}".format(activation))

def blur_padding(img, kernel=3, iteration=32, mask=None, save=False):
    # get RGB and mask
    image        = img[..., :3]
    
    if mask is None:
        mask     = img[..., 3, np.newaxis] / 255
        
    # masking
    image        = image * mask
    new_img = image.copy()
    new_msk = mask.copy()
    for _ in range(iteration):
        # kernel
        dilate_k = np.ones((kernel,kernel))
        
        # each color
        dilate_r = cv2.dilate(new_img[..., 0, np.newaxis], dilate_k, iterations=1)  # make dilation image
        dilate_g = cv2.dilate(new_img[..., 1, np.newaxis], dilate_k, iterations=1)  # make dilation image
        dilate_b = cv2.dilate(new_img[..., 2, np.newaxis], dilate_k, iterations=1)  # make dilation image
        
        # mask
        dilate_m = cv2.dilate(new_msk, dilate_k, iterations=1)  # make dilation image        
        dilate_m = dilate_m[...,np.newaxis]
        dilate_m = dilate_m - new_msk
        
        # concatenate all channel
        dilate_image = np.concatenate((
                dilate_r[...,np.newaxis], 
                dilate_g[...,np.newaxis], 
                dilate_b[...,np.newaxis]
            ),axis=2)
        
        # mask for only dilated region
        dilate_image = dilate_image * dilate_m
                
        # update dilated pixels
        new_img = new_img + dilate_image
        new_msk = new_msk + dilate_m
        
    new_img = cv2.GaussianBlur(new_img, (7, 7), 0)
    new_img = new_img * (1-mask) + image
    new_img = new_img.astype(np.uint8)
        
    if save:
        Image.fromarray(new_img).save('blurred.png')
    return new_img

def feature_size(size, kernel_size=5, stride=1, dilation=1, padding=0, num_layers=1, pool=0):
    for _ in range(num_layers):
        size = ((size + 2*padding - dilation*(kernel_size-1) - 1)/ stride) + 1
        if pool > 1:
            size = size // 2
    return int(size)

############   Helpers   #################
def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias,   0.0)

############   Generic   #################
class SkipConnect(nn.Module):
    def __init__(self, submodule=None):
        super().__init__()
        self.submodule = submodule
    def forward(self, x):
        return torch.cat([x, self.submodule(x)], dim=1)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None, output_gate=False):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        init_He(self)
        self.activation = activation
        self.output_gate = output_gate

    def forward(self, x):
        """ Out = act(Feature) * sig(Gating) """ 
        feature = self.input_conv(x)
        if self.activation:
            feature = self.activation(feature)

        gating = nn.Sigmoid()(self.gating_conv(x))
        out = feature * gating

        if self.output_gate:
            return out, gating            
        return out

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding=0, dilation=1, bias=False, weight_norm='none',
                 use_gate=False, activation=None, output_gate=False):
        super().__init__()
        
        if use_gate:
            self.conv = GatedConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias,
                              activation=activation, output_gate=output_gate)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.weight_norm = get_weightnorm(weight_norm)
        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

class Conv2dBlock(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            bias         = False,
            padding_type = 'zero',
            weight_norm  = 'none',
            norm_layer   = nn.InstanceNorm2d,
            activation   = 'none',
            use_gate     = False,
            output_gate  = False,
            fsize        = None
        ):
        super().__init__()

        self.use_gate = use_gate
        self.output_gate = output_gate

        # initialize activation
        self.activation = get_activation(activation)
        self.pad = get_padding(padding_type, padding)

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias),
                nn.Sigmoid()
            )
            init_He(self)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=0, dilation=dilation, bias=bias)
        
        self.norm = norm_layer(out_channels) if norm_layer!= None else norm_layer

        if fsize != None and norm_layer!= nn.InstanceNorm2d:
            self.norm = norm_layer(out_channels, fsize=fsize)

        self.weight_norm = get_weightnorm(weight_norm)
        if self.weight_norm != None:
            self.conv = self.weight_norm(self.conv)
        

    def forward(self, x):
        out = self.conv(self.pad(x))
        if self.norm:
            out = self.norm(out)
            
        if self.activation:
            out = self.activation(out)

        if self.use_gate:
            gate = self.gate(self.pad(x))
            if self.output_gate:
                return out, gate
            out = out * gate

        return out

class ResnetBlock(nn.Module):
    def __init__(self, 
            dim, 
            norm_layer, 
            padding_type = 'reflect',
            kernel_size  = 3, 
            dilate       = 1,
            bias         = True, 
            use_dropout  = False, 
            output_gate  = False,
        ):
        super().__init__()
        self.output_gate = output_gate
       
        block = []

        p = 0
        pad = get_padding(padding_type, dilate)
        # normal Convolution
        conv2d = nn.Conv2d(dim, dim, kernel_size, padding=p, bias=bias, dilation=dilate)

        block += [pad, conv2d, norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [pad, conv2d, norm_layer(dim)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        if self.output_gate:
            out, mask = self.block(x)
            return out + x, mask
        else:
            return x + self.block(x)

class ResnetBlock2(nn.Module):
    def __init__(self, 
            in_channels,
            out_channels= None, 
            padding_type= 'zero', 
            norm_layer  = nn.InstanceNorm2d, 
            kernel_size = 3, 
            dilate      = 1,
            bias        = True, 
            use_dropout = False, 
            use_gate    = False, 
            activation  = 'relu',
            output_gate = False,
            fsize       = 16, 
            upsample    = False,
            downsample  = False,
        ):
        super().__init__()
        self.upsample = upsample
        out_channels = out_channels if out_channels else in_channels

        S = 2 if downsample else 1

        if out_channels != in_channels or downsample:
            self.skip_connect = Conv2dBlock(in_channels, out_channels, 1, S, bias=False, norm_layer=None, activation='none') 
        else:
            self.skip_connect = None

        self.conv_block = nn.Sequential(
            Conv2dBlock(
                in_channels, 
                out_channels if upsample else in_channels, 
                kernel_size, 
                stride       = S,
                padding      = dilate, 
                dilation     = dilate,
                bias         = bias, 
                padding_type = padding_type, 
                use_gate     = use_gate, 
                norm_layer   = norm_layer, 
                activation   = activation),
            Conv2dBlock(
                out_channels if upsample else in_channels, 
                out_channels, 
                kernel_size, 
                stride       = 1,
                padding      = dilate, 
                dilation     = dilate,
                bias         = bias, 
                padding_type = padding_type, 
                use_gate     = use_gate, 
                norm_layer   = norm_layer, 
                activation   = 'none')
        )

    def forward(self, x):
        if self.upsample:
            x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        skip = self.skip_connect(x) if self.skip_connect else x
        return skip + self.conv_block(x) # add skip connections


class Encoder(nn.Module):
    def __init__(self, 
            input_nc     = 3,
            ngf          = 32,
            num_layers   = 3,
            norm_layer   = nn.InstanceNorm2d,
            padding_type = 'zero',
            use_gate     = False,
            use_dilate   = False,
            getIntermFeat= True,
            output_gate  = False,
            activation   = 'lrelu',
            opt          = None,
        ):
        super().__init__()
        self.getIntermFeat = getIntermFeat # 'texture', 'mask'
        self.num_layers = num_layers

        # normalization
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # use gated conv
        if opt:
            use_gate = opt.use_gate if not use_gate else use_gate
            ngf      = opt.ngf
            self.opt = opt
        
        if self.opt.mode == 'test' and self.getIntermFeat and not self.opt.infer:
            output_gate = True

        conv_blocks = []
        conv_blocks += [
                [Conv2dBlock(input_nc, ngf, 7, 1, 3, bias=use_bias, padding_type=padding_type, use_gate=use_gate, output_gate=output_gate, norm_layer=norm_layer, activation=activation)]
            ]

        for _ in range(self.num_layers):
            conv_blocks += [
                    [Conv2dBlock(ngf, ngf*2, 3, 2, 1, bias=use_bias, padding_type=padding_type, use_gate=use_gate, output_gate=output_gate, norm_layer=norm_layer, activation=activation)]
                ]
            ngf = ngf*2

        if self.getIntermFeat:
            for n in range(self.num_layers+1):
                setattr(self, f'conv{n}', nn.Sequential(*conv_blocks[n]))
        else:
            self.conv_blocks = nn.Sequential(*[conv for convs in conv_blocks for conv in convs])

        self.output_nc = ngf

    def forward(self, x, gc=False):
        if self.getIntermFeat:
            res = [x]
            for n in range(self.num_layers+1):
                out = getattr(self, f'conv{n}')(res[-1])

                if self.opt.mode == 'test' and not self.opt.infer:
                    out = save_feat(out, self.opt, type='enc', num=n, gc=gc)

                res.append(out)
            return res[1:] # ignore input
        else:
            return self.conv_blocks(x)

class Decoder(nn.Module):
    def __init__(self, 
            input_nc    = 256, 
            output_nc   = 3, 
            ndf         = 32, 
            num_layers  = 3,
            norm_layer  = nn.InstanceNorm2d,
            padding_type='zero',
            use_gate    = False,
            output_gate = False,
            opt         = None
        ):
        super().__init__()

        self.num_layers = num_layers
        self.no_skip    = False
        self.mask_enc   = False
        fs = 32

        if opt != None:
            self.num_layers = opt.num_layers
            self.no_skip    = opt.no_skip
            use_gate        = opt.use_gate if not use_gate else use_gate
            ndf             = opt.ngf
            self.opt        = opt

            fs = opt.data_size // (2**(opt.num_layers-1))

            if self.opt.mode == 'test' and not self.opt.infer:
                output_gate = True

        # normalization
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
       
        C = 1 if self.no_skip else 2
        nf = input_nc//(2*C)

        conv_blocks = []
        conv_blocks += [
                [nn.Upsample(scale_factor=2), # mode='bilinear'
                Conv2dBlock(input_nc, nf, 3, 1, 1, bias=use_bias, padding_type=padding_type, use_gate=use_gate, norm_layer=norm_layer, output_gate=output_gate, activation='relu', fsize=fs)]
            ]
        for l in range(self.num_layers-1, 0, -1):
            fs = fs*2
            conv_blocks += [
                    [nn.Upsample(scale_factor=2), # mode='bilinear'
                    Conv2dBlock(nf*C, nf//2, 3, 1, 1, bias=use_bias, padding_type=padding_type, use_gate=use_gate, norm_layer=norm_layer, output_gate=output_gate, activation='relu', fsize=fs)]
                ]
            nf = nf //2

        conv_blocks += [
                [Conv2dBlock(nf, output_nc, 7, 1, 3, bias=use_bias, padding_type=padding_type, use_gate=use_gate, norm_layer=None, activation='none'),
                nn.Sigmoid()]
            ]
        
        for n in range(self.num_layers+1):
            setattr(self, f'upconv{n}', nn.Sequential(*conv_blocks[n]))

    def forward(self, x, ef_list=None, cfattn=None, mask=None, gc=False):
        # ef_list : List[tensor] encoded features
        out = x
        # import pdb;pdb.set_trace()
        for i in range(self.num_layers):
            ef = ef_list[-(i+1)]

            # skip connection
            out = out if self.no_skip else torch.cat((out,  ef), dim=1)
            out = getattr(self, f'upconv{i}')(out)

            if self.opt.mode == 'test' and not self.opt.infer:
                out = save_feat(out, self.opt, type='dec', num=i, gc=gc)

        out = getattr(self, f'upconv{self.num_layers}')(out)
        return out

# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(nn.Module):
    """
    https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/models/networks.py#L385
    """
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        import torchvision.models as models
        # vgg_pretrained_features = models.vgg19(pretrained=True).features # vgg model download
    
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/source/jihyeon/Sample-Refine2/models/networks/network_checkpoint/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg19.features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
