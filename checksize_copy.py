from random import sample
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
#from src.model import final_model7
import utility
from model import final_model7, final_model2, nlsn, test2, test3, copyf_mod2, test4, new
#from model import unet_model
import torchvision
from option import args

checkpoint = utility.checkpoint(args)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dilation', type=bool, default=False, help='whether to use dilation in the convolutional layers')
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=3)
    parser.add_argument('--n_resblock', type=int, default=32)
    parser.add_argument('--dir_data', type=str, default='/home/vtrg/Desktop/Jagrati/Dataset/', #'../../Dataset/',
                    help='dataset directory')
    
    args = argparse.Namespace()
    args.input_channels = 3
    args.input_height = 128
    args.input_width = 128
    args.channels = 64
    args.output_channels = 3

    args = parser.parse_args()
    _model = test3.make_model(args)
    print(_model)
    print('Total params: %.2fM' % (sum(p.numel()
                  for p in _model.parameters())/1000000.0))
    #summary(_model,(3, 64, 64))
    from torchsummary import summary

   # model = NLSRNet(args)clear

    summary(_model, input_size=(args.channels, args.input_channels, args.output_channels))

    # model = torchvision.models.resnet50()
    # summary(model, (3, 224, 224))


