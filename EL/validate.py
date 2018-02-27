# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:10:40 2018

@author: Fan
"""
import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from torch.autograd import Variable
from Argparser import Argparser
from libs.utils  import DataNormalize,InputForm
from libs.net    import ELnet 
from Evautils    import Record,accuracy

parser = Argparser()
args = parser.parse_args()

def main():
    global args
    load_path = args.load_folder
    net = torch.load(load_path + 'ELnet.pkl')
    TestDataset = InputForm.Img(args.test_folder_NG,args.test_folder_OK,
                                transform = [DataNormalize.meanSubtract,DataNormalize.addAxis,
                                             DataNormalize.cvt2flt32,torch.Tensor],
                                rate = args.sp)
    test_loader = torch.utils.data.DataLoader(
                                              TestDataset,
                                              batch_size= args.batch_size, 
                                              shuffle=True \
                                              #,num_workers = 4
                                             )
    loss_function = nn.CrossEntropyLoss()      
    accuracy = validate(net,test_loader,loss_function)
    print('Accuracy on testset:' + str(accuracy))
    

def validate(model,dataset,loss_function):
    global args
    
    model.eval()
    accuracys = Record('accuracy',state = 'test')
    losses = Record('loss',state = 'test')
    for i,(inputs,labels) in enumerate(dataset):
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        x = Variable(inputs,volatile=True)
        y = Variable(labels,volatile=True)
        output = model(x)               
        loss   = loss_function(output, y)   
        accu   = accuracy(output.data,labels)
        losses.update(loss.data[0],inputs.size(0))
        accuracys.update(accu,inputs.size(0))

        if i % args.print_freq == 0:
            print('step :' + str(i) +'||loss: ' + str(losses.val))
            print('Avg loss: ' + str(losses.avg))
            print('accuracy: ' + str(accuracys.val))
            print('Avg accuracy: ' + str(accuracys.avg))
                
        losses.record()
        accuracy.record()
    return accuracys.avg


if __name__ == '__main__':
    main()


    