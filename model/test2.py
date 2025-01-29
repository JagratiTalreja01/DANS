import torch
import torch.nn as nn
import argparse
import torch.nn as nn
from multiprocessing import reduction
import re
from turtle import forward

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
    def __init__(self, args, in_channels, out_channels, scale_factor=2, conv = common.default_conv):
        super(NLSRUpsampler, self).__init__()

        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.res = common.ResBlock(out_channels, out_channels // 2, kernel_size = 3)
        self.nlsr = attention.NonLocalSparseAttention(out_channels, out_channels // 2)
        

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        #x = self.res(x)
        x = self.nlsr(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))
        x4 = nn.functional.relu(self.conv4(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x
        
class NLSRNet(nn.Module):
    def __init__(self, args):
        super(NLSRNet, self).__init__()

        channels = 64

        self.input_conv = nn.Conv2d(args.input_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_1 = InceptionModule(channels, 2*channels)
        self.nlsr_downsample_1 = NLSRUpsampler(2*channels, 2*channels)
        self.inception_2 = InceptionModule(2*channels, 4*channels)
        self.nlsr_downsample_2 = NLSRUpsampler(4*channels, 4*channels)
        self.inception_3 = InceptionModule(4*channels, 8*channels)
        self.nlsr_downsample_3 = NLSRUpsampler(8*channels, 8*channels)
        self.inception_4 = InceptionModule(8*channels, 16*channels)
        self.nlsr_downsample_4 = NLSRUpsampler(16*channels, 16*channels)
        self.nlsr_upsample_4 = NLSRUpsampler(16*channels, 8*channels)
        self.inception_5 = InceptionModule(16*channels, 8*channels)
        self.nlsr_upsample_3 = NLSRUpsampler(8*channels, 4*channels)
        self.inception_6 = InceptionModule(8*channels, 4*channels)
        self.nlsr_upsample_2 = NLSRUpsampler(4*channels, 2*channels)
        self.inception_7 = InceptionModule(4*channels, 2*channels)
        self.nlsr_upsample_1 = NLSRUpsampler(2*channels, channels)
        self.inception_8 = InceptionModule(2*channels, channels)
        self.output_conv = nn.Conv2d(channels, args.output_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.inception_1(x1)
        x3 = self.nlsr_downsample_1(x2)
        x4 = self.inception_2(x3)
        x5 = self.nlsr_downsample_2(x4)
        x6 = self.inception_3(x5)
        x7 = self.nlsr_downsample_3(x6)
        x8 = self.inception_4(x7)
        x9 = self.nlsr_downsample_4(x8)
        x10 = self.nlsr_upsample_4(x9)
        x11 = torch.cat([x10, x8], dim=1)
        x12 = self.inception_5(x11)
        x13 = self.nlsr_upsample_3(x12)
        x14 = torch.cat([x13, x6], dim=1)
        x15 = self.inception_6(x14)
        x16 = self.nlsr_upsample_2(x15)
        x17 = torch.cat([x16, x4], dim=1)
        x18 = self.inception_7(x17)
        x19 = self.nlsr_upsample_1(x18)
        x20 = torch.cat([x19, x2], dim=1)
        x21 = self.inception_8(x20)
        x22 = self.output_conv(x21)
        return x22      

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