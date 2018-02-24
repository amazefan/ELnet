# -*- coding: utf-8 -*-

import os
import torch
import torch.utils.data as Data
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from DataNormalize  import *


NG_root = r'E:\hongpucorp\EL\data\defect pic\NG_AUG'
OK_root = r'E:\hongpucorp\EL\data\defect pic\OK'

ELdata = Img(NG_root,OK_root,  ifrate = True,
             transform = ['meanSubtract','addAxis','cvt2flt32','torch.Tensor'],
             rate = 0.5)
train_loader = torch.utils.data.DataLoader(
                                            ELdata,
                                            batch_size=8, 
                                            shuffle=True \
                                            #,num_workers = 4
                                            )  
net = ELnet([2,2,2,2])
loss_function = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters() , lr   = 0.001)

for i, (inputs,label) in enumerate(train_loader):
    x = Variable(inputs)
    y = Variable(label)
    output = net(x)               
    loss = loss_function(output, y)   
    optimizer.zero_grad()           
    loss.backward()                
    optimizer.step()
    if i % 5 == 0:
        print('step :' + str(i) +'||loss: ' + str(loss))
        print(output)
        

class Record(object):
    
    def __init__(self,name):
        self.reset()
        self.name = name
        self.cwd = os.getcwd()
        with open(self.cwd + '\\trainRecord__' + self.name + '.txt','w') as txt:
            txt.write('\n')
            txt.close()
        
    def reset(self):
        self.val = 0
        self.cnt = 0
        self.avg = 0
        self.sum = 0
    def update(self , val , num):
        self.val = val
        self.sum += val*num
        self.cnt += num
        self.avg = self.sum/self.cnt 
    def record(self):
        self.dir = {self.name +'_val':self.val,self.name + '_avg':self.avg}
        with open(self.cwd + '\\trainRecord__' + self.name + '.txt','r+') as txt:
            txt.read()
            txt.write('\n'+str(self.dir))
            txt.close()
            
def accuracy(output , label):
    _,pred_index = output.topk(1,1)            
    batch_size = label.size(0)
    true_num = 0
    for i in range(batch_size):
        if label[i][pred_index[i][0]] == 1:
            true_num +=1
    return 100*true_num/batch_size
         
            
            