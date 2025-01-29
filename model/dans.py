#Run this model for Image super resolution with input image size(64,64,3)

import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn

from model import common
#import common
from model import attention


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return NLSRNet(args, dilated.dilated_conv)
    else:
        return NLSRNet(args)

class NLSRUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(NLSRUpsampler, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nlsr = attention.NonLocalSparseAttention(out_channels, out_channels // 2)
        self.conv_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.nlsr(x) #[2, 128,128,128]
        #x = self.conv_transpose(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_res = self.conv_res(x)
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))
        x4 = nn.functional.relu(self.conv4(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x += x_res
        x += x
        return nn.functional.relu(x)
        
class NLSRNet(nn.Module):
    def __init__(self, args):
        super(NLSRNet, self).__init__()
        channels = args.channels
        self.input_conv = nn.Conv2d(args.input_channels, channels, kernel_size=3, stride=1, padding=1, bias=True) #[2,3, 64, 64] ##ideal 3x3 Conv, 64: (64, 64, 64)
        self.inception_1 = InceptionModule(channels, 2*channels) #[2, 64, 64, 64]   # idealInception Module: (256, 64, 64)
        self.nlsr_downsample_1 = NLSRUpsampler(2*channels, 2*channels) #[2, 128, 64, 64] # ideal NLSR Upsampler, 2*64: (128, 128, 64)
        self.inception_2 = InceptionModule(2*channels, 4*channels)                       # ideal Inception Module: (256, 128, 128)
        self.nlsr_downsample_2 = NLSRUpsampler(4*channels, 4*channels)                   # ideal NLSR Upsampler, 4*64: (256, 256, 64)
        self.inception_3 = InceptionModule(4*channels, 8*channels)                       # ideal Inception Module: (256, 256, 256)
        self.nlsr_downsample_3 = NLSRUpsampler(8*channels, 8*channels)                   # ideal NLSR Upsampler, 8*64: (512, 512, 64)
        self.inception_4 = InceptionModule(8*channels, 16*channels)                      # ideal Inception Module: (256, 512, 512)
        self.nlsr_downsample_4 = NLSRUpsampler(16*channels, 16*channels)                 # ideal NLSR Upsampler, 16*64: (1024, 1024, 64)
        self.nlsr_upsample_4 = NLSRUpsampler(16*channels, 8*channels)                    # ideal NLSR Upsampler, 8*64: (512, 512, 64)
        self.inception_5 = InceptionModule(16*channels, 8*channels)                      # ideal Inception Module: (256, 512, 512)
        self.nlsr_upsample_3 = NLSRUpsampler(8*channels, 4*channels)                     # ideal NLSR Upsampler, 4*64: (256, 256, 64)
        self.inception_6 = InceptionModule(8*channels, 4*channels)                       # ideal Inception Module: (256, 256, 256)
        self.nlsr_upsample_2 = NLSRUpsampler(4*channels, 2*channels)                     # ideal NLSR Upsampler, 2*64: (128, 128, 64)
        self.inception_7 = InceptionModule(4*channels, 2*channels)                       # ideal Inception Module: (256, 128, 128)
        self.nlsr_upsample_1 = NLSRUpsampler(2*channels, channels)                       # ideal 3x3 Conv, Cout: (Cout, 128, 128)
        self.inception_8 = InceptionModule(2*channels, channels)                         #ideal Output: (Cout, 128, 128)
        self.output_conv = nn.Conv2d(channels, args.output_channels, kernel_size=3, stride=1,padding=1, bias=True) 

    def forward(self, x):
        x = self.input_conv(x) #[2,3, 64, 64]
        x = self.inception_1(x) #[2, 64, 64, 64]
        x = self.nlsr_downsample_1(x) #[2, 128, 64, 64]
        x = self.inception_2(x)
        x = self.nlsr_downsample_2(x)
        x = self.inception_3(x)
        x = self.nlsr_downsample_3(x)
        x = self.inception_4(x)
        x = self.nlsr_downsample_4(x)
        x = self.nlsr_upsample_4(x)
        x = self.inception_5(x)
        x = self.nlsr_upsample_3(x)
        x = self.inception_6(x)
        x = self.nlsr_upsample_2(x)
        x = self.inception_7(x)
        x = self.nlsr_upsample_1(x)
        x = self.inception_8(x)
        x = self.output_conv(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
