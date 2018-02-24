# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:16:54 2018

@author: Administrator
"""

import cv2
import numpy as np
from os import path as osp
import os

#Do MeanSubtract on the input image, in order to reduce the correlation
def meanSubtract(img):
    imgNew = img.copy()
    meanPixel = np.mean(img)
    imgNew = img - meanPixel
    imgNew = imgNew.astype(np.float32)
    return imgNew

#Scaling the input image into the same size
def scaling(img,targetsize):
    h,w = img.shape
    fh = float(targetsize)/float(h)
    fw = float(targetsize)/float(w)
    im = cv2.resize(img,None,None,fx = fw ,fy = fh,
                    interpolation=cv2.INTER_LINEAR)
    return im

def addAxis(img):
    return img[np.newaxis]

def cvt2flt32(img):
    return img.astype(np.float32)

