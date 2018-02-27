# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:29:32 2018

@author: Fan
"""

import argparse
def Argparser():
    parser = argparse.ArgumentParser(description='PyTorch ELnet')
    #train argument
    parser.add_argument('--arch','-a',default = 'ResNet')
    parser.add_argument('--cuda','-c',default = True)
    parser.add_argument('--layer',nargs = 4, type = int,
                        help = 'Define layes in the Net')
    parser.add_argument('--epoches',type = int ,default = 1000,
                        help = 'Define train epoches ')
    parser.add_argument('--lr' , type = float ,default = 0.01,
                        help = 'Define origin learning rate')
    parser.add_argument('--batch_size' , type = int , default = 100,
                        help = 'Define train batch size')
    parser.add_argument('--train_folder_NG' , type = str , default = r'E:\hongpucorp\EL\data\defect pic\NG_AUG',
                        help = 'Load the train data with folder path(NG folder)')
    parser.add_argument('--train_folder_OK' , type = str , default = r'E:\hongpucorp\EL\data\defect pic\OK',
                        help = 'Load the train data with folder path(OK folder)')
    parser.add_argument('--save_folder' , type = str , default = '',
                        help = 'Folder path to save the model and trainRecord')
    parser.add_argument('--print_freq' , type = int , default = 100,
                        help = 'Frequency to print the train info(loss and accuarcy)')
    parser.add_argument('--checkpoint' , type = int , default = 100,
                        help = 'Frequency to save the model')
    parser.add_argument('--sp' , type = float , default = 0.5,
                        help = 'Sample proportion of positive via negative data(default 0.5).')
    #test argument
    parser.add_argument('--test_folder_OK' , type = str, default = '',
                        help = 'Load the test data with folder path(OK folder)')
    parser.add_argument('--test_folder_NG' , type = str, default = '',
                        help = 'Load the test data with folder path(NG folder)')
    parser.add_argument('--load_folder' , type = str , default = '',
                        help = 'Folder path to load the model')
    
    return parser

