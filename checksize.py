from random import sample
import torch
import torch.nn as nn
from torchsummary import summary
#from src.model import copyf_mod2
import utility
from model import nlsn, final_model2,test3, jagrati_model
#from model import unet_model
import torchvision
from option import args

checkpoint = utility.checkpoint(args)

if __name__ == "__main__":
    
    _model = jagrati_model.make_model(args)
    print(_model)
    print('Total params: %.2fM' % (sum(p.numel()
                  for p in _model.parameters())/1000000.0))
    summary(_model,(3, 256, 256))

    # model = torchvision.models.resnet50()
    # summary(model, (3, 224, 224))