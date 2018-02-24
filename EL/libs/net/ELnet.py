# -*- coding: utf-8 -*-



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicBlock import *


class ELnet(nn.Module):
    def __init__(self,layer,conv_num = [2,2,2,2],num_class=2):
        super(ELnet,self).__init__()
        initalLayer = con3x3(1,64)
        initalLayer.append(nn.MaxPool2d(2))
        self.layer0 = nn.Sequential(
                                    *initalLayer
                                    )
        self.layer1 = self._buildblock(64,64,layer[0],conv_num[0],pooling=False)
        self.layer2 = self._buildblock(64,128,layer[1],conv_num[1])
        self.layer3 = self._buildblock(128,256,layer[2],conv_num[2])
        self.layer4 = self._buildblock(256,512,layer[3],conv_num[3])
        self.avgpool = nn.AvgPool2d(24)
        self.fc = nn.Linear(512 , num_class)
        
    def _buildblock(self,in_channels,channels,block_num,
                    conv_num = 2 ,pooling = True ):
        layers = [resBlock(in_channels,channels,conv_num,pooling)]
        for i in range(block_num-1):
            layers.append(resBlock(channels,channels,conv_num,pooling = False))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
    

    
    
    
    
    
    
    
    