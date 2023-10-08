import torch
import torch.nn as nn

from utils.utils import *

# Downsampling
def ConvBR(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, bn=True, leaky=False): 
    module = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)]
    if bn:
        module += [nn.BatchNorm2d(out_channels)]
    module += [nn.LeakyReLU(0.2, True)] if leaky else [nn.ReLU(True)]
    return nn.Sequential(*module)

# Upsampling
def deConvBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class UNetGeneratorT2S(nn.Module):  
    def __init__(self, input_nc=3, output_nc=3, ngf=64): 
        super().__init__() 

        # Contracting path
        self.enc1 = ConvBR(in_channels=input_nc, out_channels=ngf, bn=False)
        self.enc2 = ConvBR(in_channels=ngf,   out_channels=ngf*2)
        self.enc3 = ConvBR(in_channels=ngf*2, out_channels=ngf*4)
        self.enc4 = ConvBR(in_channels=ngf*4, out_channels=ngf*8)
        self.enc5 = ConvBR(in_channels=ngf*8, out_channels=ngf*8)
        self.enc6 = ConvBR(in_channels=ngf*8, out_channels=ngf*8)
        self.enc7 = ConvBR(in_channels=ngf*8, out_channels=ngf*8)
          
        # Expansive path
        self.dec7 = deConvBR(in_channels=ngf*8,   out_channels=ngf*8)
        self.dec6 = deConvBR(in_channels=ngf*8*2, out_channels=ngf*8) # concatenated input: enc6
        self.dec5 = deConvBR(in_channels=ngf*8*2, out_channels=ngf*8) # concatenated input: enc5
        self.dec4 = deConvBR(in_channels=ngf*8*2, out_channels=ngf*4) # concatenated input: enc4
        self.dec3 = deConvBR(in_channels=ngf*4*2, out_channels=ngf*2) # concatenated input: enc3
        self.dec2 = deConvBR(in_channels=ngf*2*2, out_channels=ngf)  # concatenated input: enc2

        # self.dec1 = deConvBR(in_channels=64*2, out_channels=32)    # concatenated input: enc1
        # self.dec0 = deConvBR(in_channels=64, out_channels=32)
        
        self.dec1 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=ngf*2, out_channels=output_nc, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid() ## was not in the Tex2Shape. it is for clamping values to be in 0 to 1
            )
  
    def forward(self, x):
        # Downsampling                            1    3  512  512
        enc1 = self.enc1(x)    # 3 -> 64        ([1,  64, 256, 256])        
        enc2 = self.enc2(enc1) # 64 -> 128      ([1, 128, 128, 128])
        enc3 = self.enc3(enc2) # 128 -> 256     ([1, 256,  64,  64])
        enc4 = self.enc4(enc3) # 256 -> 512     ([1, 512,  32,  32])
        enc5 = self.enc5(enc4) # 512 -> 512     ([1, 512,  16,  16])
        enc6 = self.enc6(enc5) # 512 -> 512     ([1, 512,   8,   8])
        enc7 = self.enc7(enc6) # 512 -> 512     ([1, 512,   4,   4])

        # Upsampling
        dec7 = self.dec7(enc7) # 512 -> 512                  # ([1,  512,   8,   8])
        cat7 = torch.cat((dec7, enc6), dim= 1) # 512 (+512) -> ([1, 1024,   8,   8])

        dec6 = self.dec6(cat7) # 1024 -> 512                 # ([1,  512,  16,  16])
        cat6 = torch.cat((dec6, enc5), dim= 1) # 512 (+512) -> ([1, 1024,  16,  16])

        dec5 = self.dec5(cat6) # 1024 -> 512                 # ([1,  512,  32,  32])
        cat5 = torch.cat((dec5, enc4), dim= 1) # 512 (+512) -> ([1, 1024,  32,  32])

        dec4 = self.dec4(cat5) # 1024 -> 256                 # ([1,  256,  64,  64])
        cat4 = torch.cat((dec4, enc3), dim= 1) # 256 (+256) -> ([1,  512,  64,  64])

        dec3 = self.dec3(cat4) # 512 -> 128                  # ([1,  128, 128, 128])
        cat3 = torch.cat((dec3, enc2), dim= 1) # 128 (+128) -> ([1,  256, 128, 128])

        dec2 = self.dec2(cat3) # 256 -> 64                   # ([1,   64, 256, 256])
        cat2 = torch.cat((dec2, enc1), dim= 1) # 64 (+64)   -> ([1,  128, 256, 265])

        dec1 = self.dec1(cat2) # 128 -> 3                   -> ([1,    3, 512, 512])
        return dec1 
    
    def summary(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class UNetGenerator360(nn.Module):  
    def __init__(self, 
        input_nc    = 6, 
        output_nc   = 3,
        norm_layer  = nn.BatchNorm2d,
        ):
        super().__init__() 

        # normalization
        use_bias = norm_layer == nn.InstanceNorm2d
        padding = nn.ReflectionPad2d

        self.enc1 = nn.Sequential(
            padding(2),
            nn.Conv2d(input_nc, 32,  kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )
        
        self.res_block0 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block1 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block2 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block3 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block4 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block5 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block6 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        self.res_block7 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(256),
        )
        
        self.concatenate = True
        C = 2 if self.concatenate else 1

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256*C, 128,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128*C, 64,   kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64*C,  32,   kernel_size=4, stride=2, padding=0, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32,output_nc,kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )
      
    def forward(self, x, mask=None, GT=None):
        
        enc1 = self.enc1(x)     #254
        enc2 = self.enc2(enc1)  #126
        enc3 = self.enc3(enc2)  #62
        enc4 = self.enc4(enc3)  #30

        resB = self.res_block0(enc4) + enc4
        resB = self.res_block1(resB) + resB
        resB = self.res_block2(resB) + resB
        resB = self.res_block3(resB) + resB
        resB = self.res_block4(resB) + resB
        resB = self.res_block5(resB) + resB
        resB = self.res_block6(resB) + resB
        resB = self.res_block7(resB) + resB
        # import pdb;pdb.set_trace()

        cat1 = torch.cat((resB, enc4), dim=1) if self.concatenate else resB + enc4
        dec1 = self.dec1(cat1)  #62
        
        cat2 = torch.cat((dec1, enc3), dim=1) if self.concatenate else dec1 + enc3
        dec2 = self.dec2(cat2)  #126

        cat3 = torch.cat((dec2, enc2), dim=1) if self.concatenate else dec2 + enc2
        dec3 = self.dec3(cat3)  #254

        out = self.dec4(dec3)   #256
        return out
    
    def summary(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
            
class PatchDiscriminatorT2S(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        # input = 
        # partial + generated maps or partial + GT maps = torch.cat( (512, 512, 3), (512, 512, 3) )
        # if generated map only -> (1,  3,  512, 512)   else (1,   6, 512, 512) 
        self.conv1 = ConvBR(in_channels=input_nc, out_channels=ndf,  bn=False)     # (1,  64, 256, 256)
        self.conv2 = ConvBR(in_channels=ndf,      out_channels=ndf*2)              # (1, 128, 128, 128)
        self.conv3 = ConvBR(in_channels=ndf*2,    out_channels=ndf*4)              # (1, 256,  64,  64)
        self.conv4 = ConvBR(in_channels=ndf*4,    out_channels=ndf*8)              # (1, 512,  32,  32)
        self.feature = nn.Sequential(                                              # (1,   1,  30,  30)
            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=5, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.feature(x)
        # sigmoid => use BCEWithLogitsLoss in calculating loss
        return out

    def summary(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

class PatchDiscriminator360(nn.Module):
    def __init__(self,
        input_nc    = 6,
        norm_layer  = nn.BatchNorm2d,
        ):
        super().__init__() 

        # normalization
        use_bias = norm_layer == nn.InstanceNorm2d
        padding = nn.ReflectionPad2d

        self.conv1 = nn.Sequential(
            padding(3),
            nn.Conv2d(input_nc, 128,  kernel_size=7, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            padding(1),
            nn.Conv2d(128, 256,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )
        self.conv3 = nn.Sequential(
            padding(1),
            nn.Conv2d(256, 512,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            padding(1),
            nn.Conv2d(512, 1024,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            padding(1),
            nn.Conv2d(1024, 2048,  kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )
        self.conv6 = nn.Sequential(
            padding(1),
            nn.Conv2d(2048, 1,  kernel_size=4, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)
        # sigmoid => use BCEWithLogitsLoss in calculating loss
        return out

    def summary(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())