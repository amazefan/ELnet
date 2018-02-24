# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:54:19 2018

@author: Administrator
"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = r'E:\python.file\EL'

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'libs')
add_path(lib_path)

