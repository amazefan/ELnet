# -*- coding: utf-8 -*-


"""
Convert the image into net input form
"""

import cv2
import numpy as np
import os
import glob
import torch
import torch.utils.data as Data
from os import path as osp
from DataNormalize import scaling

'''
Cache Method(Not used)
For images are too big to fill the cache.(BOOM)
'''
#DataSet in form : NG folder stores the positive samples.
#                  OK folder stores the negative samples. 
def GetImgAndLabel(folder , labelIndex):
    imgs = glob.glob(folder + '/*.jpg')
    
    imgs_Array = []
    for imgPath in imgs:
        img = meanSubtract(cv2.imread(imgPath,0))
        img.astype(np.float32)
        #img = img[:,:,np.newaxis]
        imgs_Array.append(img)
    imgs_Array = imgs_Array[1:]
    num = len(imgs_Array)
    if labelIndex:
        imgs_Label = np.ones((num))*labelIndex
    else:
        imgs_Label = np.zeros((num))
    imgs_Label.astype(np.int64)
    #imgs_Array = np.asarray(imgs_Array)
    return imgs_Array,imgs_Label

#Combine the positive and negative samples' data and labels
#Rate means: positive:negative ratio
def CombinePF(NG_folder,OK_folder,ifrate = False,rate = None):
    imgs_NG_Array , imgs_NG_Label = GetImgAndLabel(NG_folder,1)
    imgs_OK_Array , imgs_OK_Label = GetImgAndLabel(OK_folder,0)
    
    if ifrate:
        NG_num = imgs_NG_Label.shape
        OK_num_R = int(float(NG_num)/rate)
        imgs_OK_Array = np.random.choice(imgs_OK_Array, OK_num_R)
        imgs_OK_Label = np.zeros((OK_num_R)).astype(np.int64)
        
    imgs_Array = np.asarray(imgs_NG_Array + imgs_OK_Array)
    imgs_Label = np.hstack((imgs_NG_Label,imgs_OK_Label))
    
    train_set_x = torch.FloatTensor(imgs_Array)
    train_set_y =torch.LongTensor(imgs_Label)
    
    torch_dataset = Data.TensorDataset(data_tensor = train_set_x, target_tensor = train_set_y)
    return torch_dataset

'''
Iter Method:Quickly and lighter
'''
#Create a list to store image pathes and labels
#Release the Cache
def imgList(NG_folder,OK_folder,ifrate = False ,rate = None):
    imgs_NG_path = glob.glob(NG_folder + '/*.jpg')
    imgs_OK_path = glob.glob(OK_folder + '/*.jpg')
    
    imgs_list = [(path,torch.Tensor((1,0))) for path in imgs_NG_path]
    
    if ifrate:
        NG_num = len(imgs_NG_path)
        OK_num_R = int(float(NG_num)/rate)
        imgs_OK_path = np.random.choice(imgs_OK_path,OK_num_R).tolist()
    
    imgs_list.extend([(path,torch.Tensor((0,1))) for path in imgs_OK_path])
    return imgs_list
        
#Define a class based on torch.utils.data.dataset to get input samples
class Img(Data.Dataset):
    def __init__(self,NG_root,OK_root, defaultLoader = imgList, scale = True,
                 scalesize = 400 , transform = None , ifrate = False , rate = None):
        self.transform = transform
        self.scale = scale
        self.scalesize = scalesize
        self.imgList = imgList(NG_root,OK_root,ifrate,rate)

    def __getitem__(self,index):
        imgPath , label = self.imgList[index]
        img = cv2.imread(imgPath,0)
        
        if self.scale is True:
            img = scaling(img,self.scalesize)
        if self.transform is not None:
            for method in self.transform:
                img = eval(method)(img)
                
        return img, label
        
    def __len__(self):
        return len(self.imgList)
        
    def getsize(self,index):
        imgPath , label= self.imgList[index]
        img = cv2.imread(imgPath,0)
        h,w = img.shape
        return h,w
        
        

