# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:22:20 2018

@author: Administrator
"""

img_path = r'E:\hongpucorp\EL\data\defect pic\xuhan'
for files in os.listdir(img_path):
    if files.split('.')[-1] == 'xml':
        img_file = img_path + '\\' + files.split('.')[0] +'.jpg'
        xml_file = img_path + '\\' + files
        cutimg(img_file,xml_file)
