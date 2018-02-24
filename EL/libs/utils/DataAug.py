# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:56:47 2018

@author: Administrator
"""

import cv2
import numpy as np
from os import path as osp
import glob

#Do dataAug to the positive samples, use 3 modes recently.
def dataAug(img_path,save_dir = r'E:\hongpucorp\EL\data\defect pic\NG_AUG\\'):
    img = cv2.imread(img_path,0)
    Augmode1 = img.copy()
    Augmode2 = img.copy()
    Augmode3 = img.copy()
    
    Augmode1 = cv2.flip(img, 0)
    Augmode2 = cv2.flip(img, 1)
    Augmode3 = cv2.flip(img,-1)
    
    basename = osp.basename(img_path)
    basename = osp.splitext(basename)[0]
    
    cv2.imwrite(save_dir + basename + '_o' + '.jpg', img)
    cv2.imwrite(save_dir + basename + '_h' + '.jpg', Augmode1)
    cv2.imwrite(save_dir + basename + '_v' + '.jpg', Augmode2)
    cv2.imwrite(save_dir + basename + '_b' + '.jpg', Augmode3)

#Carry out dataAug to image dir    
def AugtoAll(base_dir):
    imgs = glob.glob(base_dir + '/*.jpg')
    #map(dataAug,imgs)         WHY CAN'T?
    for img in imgs:
        dataAug(img)


