# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#define basic convolutional filter function
#Args include:
#             in_channels : the channel dimension of input Tensor
#             out_channels: the channel dimension of filted Tensor
#             stride      : the stride filter moves
#             relu        : whether to carry out relu in the tail part of the con3x3 module
#             pad         : should be 0/1,1 means pad the filter image Tensor,
#                                         0 means not pad the filter image Tensor
def con3x3(in_channels , out_channels , stride = 1 , relu = True, pad = 1):
    conv = nn.Conv2d(in_channels , out_channels , kernel_size = 3,
                     stride = stride , padding = pad , bias = False)
    bn   = nn.BatchNorm2d(out_channels)
    Relu = nn.ReLU(inplace=True)
    if relu:
        module = [conv,bn,Relu]
    else:
        module = [conv,bn]
    return module

#define a ResNet basic block class
#Args include:
#             conv_nums: the number of con3x3 module included in the block
#             pooling  : whether to carray out pooling when enter the block    
class resBlock(nn.Module):
    def __init__(self,in_channels,channels,conv_nums=2,pooling = True):
        super(resBlock,self).__init__()
        assert type(conv_nums)==int and conv_nums >= 2
        self.pooling = pooling
        self.resblock = self._stacklayer(conv_nums,in_channels,channels)
    
    def _stacklayer(self,conv_nums,in_channels,channels):
        if self.pooling:
            layers = con3x3(in_channels,channels,stride = 2 ,pad=0)
        else:
            layers = con3x3(in_channels,channels)
        if conv_nums == 2:
            layers += con3x3(channels,channels,relu=False)
            resblock = nn.Sequential(
                                      *layers
                                     )
        else:
            for i in range(conv_nums-1):
                if i == conv_nums-1-1:
                    layers.extend(con3x3(channels,channels,relu=False))
                else:
                    layers.extend(con3x3(channels,channels))
            resblock = nn.Sequential(*layers)
        return resblock
        
    def forward(self,x):
        if self.pooling:
            out = self.resblock(x)
            out = F.relu(out)
            
        else:
            res = x
            
            out = self.resblock(x)
            out += res
            out = F.relu(out)
        
        return out
        
#ROIpooling: not used yet            
def ROIpooling(variable_data , w =7 , h =7):
    pass    