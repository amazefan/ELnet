# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:10:22 2018

@author: Administrator
"""

import cv2
import os 
import numpy as np
import xml.dom.minidom as minidom


def getcentroid(xml_path):
    parsexml = minidom.parse(xml_path)
    root = parsexml.documentElement
    nodes_length = len(root.getElementsByTagName('object'))
    all_centorid = np.array((0,0))
    names = []
    for i in range(nodes_length):
        xmin = eval(root.getElementsByTagName('xmin')[i].firstChild.data)
        ymin = eval(root.getElementsByTagName('ymin')[i].firstChild.data)
        xmax = eval(root.getElementsByTagName('xmax')[i].firstChild.data)
        ymax = eval(root.getElementsByTagName('ymax')[i].firstChild.data)
        name = root.getElementsByTagName('name')[i].firstChild.data
        centorid_x = int((xmax + xmin)/2)
        centorid_y = int((ymax + ymin)/2)
        centorid = np.array((centorid_y,centorid_x))
        all_centorid = np.vstack((all_centorid,centorid))
        names.append(name)
    return all_centorid,names

def cutshadow(img):
    h,w = img.shape
    thres,img2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    index = np.where(img2 == 255)
    ymin , xmin = np.min(index,axis = 1)
    ymax , xmax = np.max(index,axis = 1)
    r_sum = np.sum(img2,axis = 1)
    ymax_t = np.where(r_sum>.3*255*img.shape[1])[0][-1]
    newimg = img[ymin:ymax_t,xmin:xmax]
    return newimg,np.array((ymin,xmin))

def getimg(img_path):
    img_name = os.path.basename(img_path)
    imgname = img_name.split(r'.')[0]
    img = cv2.imread(img_path,0)
    return img,imgname

def howmanylines(img):
    thres,img2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    h,w = img2.shape
    img2 = img2[:int(h/2)]
    index = np.where((img2==0).all(0))
    var = 3
    if index[0][0]<10 and img2.shape[1]-index[0][-1]<10:
        var = 1
    elif index[0][0]<10 or img2.shape[1]-index[0][-1]<10:
        var = 2
    if index[0][0]<10:
        var-=1
    num = np.where(index[0][1:]-index[0][:-1]>50)[0].size + var

    if num<=10:
        num=10
    else:
        num=12
    return num
    
def cutimg(img_path,xml_path):
    img,imgname = getimg(img_path)
    img,bias = cutshadow(img)
    all_centorid,names = getcentroid(xml_path)
    all_centorid = all_centorid-bias
    h,w = img.shape
    h_stride = int(h/6)
    num_lines = howmanylines(img)
    w_stride = int(w/num_lines)
    lat = np.array((h_stride,w_stride))
    lat_pos = all_centorid//lat
    lat_pos = lat_pos[1:]
    for i in range(0,6):
        for j in range(0,num_lines):
            img_piece = img[i*h_stride:(i+1)*h_stride,j*w_stride:(j+1)*w_stride]
            loc = (i,j)
            if  any((loc==lat_pos).all(1)):
                os.chdir(r'E:\hongpucorp\EL\data\defect pic\NG')
                cv2.imwrite(imgname+' '+str(i)+' '+str(j)+'.jpg',img_piece)
            else:
                os.chdir(r'E:\hongpucorp\EL\data\defect pic\OK')
                cv2.imwrite(imgname+' '+str(i)+' '+str(j)+'.jpg',img_piece)
    
                
    
def howmanyxml(file_dic):
    num = 0
    for file in os.listdir(file_dic):
        if os.path.basename(file).split('.')[-1] == 'xml':
            num+=1
    return num


'''
img_path = r'E:\hongpucorp\XL\data\defect pic\xuhan'
for files in os.listdir(img_path):
    if files.split('.')[-1] == 'jpg':
        file = img_path + '\\' + files      
        img,imgname = getimg(file)
        newimg , bias = cutshadow(img)
        num = howmanylines(newimg)
        print(num)
'''    

def drawimg(img_path,xml_path):
    img,imgname = getimg(img_path)
    img,bias = cutshadow(img)
    all_centorid,names = getcentroid(xml_path)
    all_centorid = all_centorid-bias
    h,w = img.shape
    h_stride = int(h/6)
    num_lines = howmanylines(img)
    w_stride = int(w/num_lines)
    lat = np.array((h_stride,w_stride))
    lat_pos = all_centorid//lat
    lat_pos = lat_pos[1:]
    img_c = img.copy()
    img_c = cv2.cvtColor(img_c,cv2.COLOR_GRAY2BGR)
    for i in range(0,6):
        for j in range(0,num_lines):
            img_loc = img[i*h_stride:(i+1)*h_stride,j*w_stride:(j+1)*w_stride]
            loc = (i,j)
            if  any((loc==lat_pos).all(1)):               
                cv2.rectangle(img_c,(j*w_stride,i*h_stride),((j+1)*w_stride,(i+1)*h_stride) , (0,0,255), 6)
                loc = (j*w_stride+ int(1/2*w_stride),i*h_stride+ int(1/2*h_stride))
                cv2.putText(img_c,names[0],loc,cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),3)
                names.pop(0)
    os.chdir(r'E:\hongpucorp\EL\data\defect pic\drawback')
    cv2.imwrite(imgname + '_labeled' + '.jpg',img_c)




