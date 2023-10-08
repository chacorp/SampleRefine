import functools
import torch.nn as nn

from models.networks.base_network import BaseNetwork
from models.networks.architecture import *

from utils.utils import *

class PatchDiscriminator(BaseNetwork):
    def __init__(self, 
            input_nc      = 6, 
            ndf           = 64, 
            n_layers      = 4,
            kernel_size   = 4,
            norm_layer    = nn.InstanceNorm2d,
            weight_norm   = 'none',
            use_sigmoid   = False,
            getIntermFeat = False,
            opt           = None
        ):
        super(PatchDiscriminator, self).__init__()

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.opt = opt
        if self.opt != None:
            getIntermFeat  = True # False if self.opt.no_ganFeat_loss or self.opt.lambda_feat == 0 else True
            weight_norm    = self.opt.weight_norm
            n_layers       = self.opt.n_layers_D
            ndf            = self.opt.ndf
        self.getIntermFeat = getIntermFeat
        
        kw = kernel_size # 4
        padw = 1

        ## [0]
        sequence = [[
            Conv2dLayer(input_nc, ndf, kw, 2, padw, weight_norm=weight_norm), nn.LeakyReLU(0.2, True)
        ]]

        ## [1] ~ [(n_layers-1)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                Conv2dLayer(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, weight_norm=weight_norm, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]]

        ## [n_layers] == [-2]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            Conv2dLayer(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw, weight_norm=weight_norm, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        ## last [-1]
        sequence += [[
            Conv2dLayer(ndf * nf_mult, 1, kw, 1, padw, weight_norm=weight_norm)
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        self.n_layers = len(sequence)
        if getIntermFeat:
            for n in range(self.n_layers):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(self.n_layers):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

        self.init_weights(init_type='normal', gain=0.02)

    def forward(self, x):
        if self.getIntermFeat:
            res = [x]
            for n in range(self.n_layers):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, input_nc, opt, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.opt = opt
        self.num_D = opt.num_D
        self.getIntermFeat = not self.opt.no_ganFeat_loss or not (self.opt.lambda_feat == 0)
     
        # for i in range(self.num_D):
        for i in range(2): 
            netD = PatchDiscriminator(
                input_nc      = input_nc, 
                ndf           = self.opt.ndf, 
                n_layers      = self.opt.n_layers_D,
                norm_layer    = norm_layer, 
                weight_norm   = self.opt.weight_norm,
                use_sigmoid   = use_sigmoid, 
                getIntermFeat = self.getIntermFeat,
                opt           = self.opt,
                )
            
            self.n_layers = netD.n_layers
            
            if self.getIntermFeat:
                for j in range(self.n_layers):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.init_weights(init_type='normal', gain=0.02)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        # print(self)
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(i)+'_layer'+str(j)) for j in range(self.n_layers)]
            else:
                model = getattr(self, 'layer'+str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
