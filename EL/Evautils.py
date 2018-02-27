# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:24:42 2018

@author: Fan
"""
import torch

class Record(object):
    
    def __init__(self,name,state = 'train'):
        global args
        self.reset()
        self.name = name
        self.state = state
        #self.cwd = os.getcwd() 
        self.cwd = args.save_folder
        with open(self.cwd + '/' + self.state + 'Record__' + self.name + '.txt','w') as txt:
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
        with open(self.cwd + '/' + self.state + 'Record__' + self.name + '.txt','r+') as txt:
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