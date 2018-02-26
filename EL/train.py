# -*- coding: utf-8 -*-


import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from torch.autograd import Variable
from Argparser import Argparser
from libs.utils  import DataNormalize,InputForm
from libs.net    import ELnet 

parser = Argparser()
args = parser.parse_args()
def main():    
    global args   
    NG_root = args.train_folder_NG
    OK_root = args.train_folder_OK
    
    ELdata = InputForm.Img(NG_root,OK_root,  ifrate = True,
                 transform = [DataNormalize.meanSubtract,DataNormalize.addAxis,
                              DataNormalize.cvt2flt32,torch.Tensor],
                 rate = 0.5)
    train_loader = torch.utils.data.DataLoader(
                                                ELdata,
                                                batch_size= args.batch_size, 
                                                shuffle=True \
                                                #,num_workers = 4
                                                )  
    net = ELnet.ELnet(args.layer)
    loss_function = nn.CrossEntropyLoss()  
    if args.cuda:
        net = net.cuda()
        loss_function = loss_function.cuda()
    
    optimizer = torch.optim.Adam(net.parameters() , lr   = args.lr)
    epoches = args.epoches
    train(train_loader, net, loss_function, optimizer, epoches, 
          args.cuda, args.print_freq)
    cwd = args.save_folder
    torch.save(net, cwd + '\\ELnet.pkl')

def train(loader, model, loss_function, optimizer, epoches, ifcuda, print_freq):
    
    global args
    losses   = Record('losses')
    accuracys = Record('accuracy')
    
    for epoch in range(epoches):
        print('epoch: ' + str(epoch))
        if epoch%args.checkpoint == 0:
            torch.save(model,args.save_folder + '/ELnet.pkl')
        lrUpdate(optimizer,epoch) 
        for i, (inputs,labels) in enumerate(loader):
            if ifcuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            x = Variable(inputs)
            y = Variable(labels)
            output = model(x)               
            loss   = loss_function(output, y)   
            accu   = accuracy(output.data,labels)
            losses.update(loss.data[0],inputs.size(0))
            accuracys.update(accu,inputs.size(0))
           
            optimizer.zero_grad()           
            loss.backward()                
            optimizer.step()
            if i % print_freq == 0:
                print('step :' + str(i) +'||loss: ' + str(losses.val))
                print('Avg loss: ' + str(losses.avg))
                print('accuracy: ' + str(accuracys.val))
                print('Avg accuracy: ' + str(accuracys.avg))
        losses.record()
        accuracys.record() 

class Record(object):
    
    def __init__(self,name):
        global args
        self.reset()
        self.name = name
        #self.cwd = os.getcwd() 
        self.cwd = args.save_folder
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
        if label[i]==pred_index[i][0] :
            true_num +=1
    return 100*true_num/batch_size

def lrUpdate(optimizer,epoch):
    global args
    scale = np.e/10*2
    step  = 10
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    main()

            
            