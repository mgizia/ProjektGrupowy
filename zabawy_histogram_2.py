# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:25:14 2020

@author: emilk
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('test1.png',0)
#mask = cv2.imread('roboczy/maska_thresh.png',0)

def RysujHistogram(img_path, mask_path, save_folder='roboczy'):

    img = cv2.imread(img_path,0)
    mask = cv2.imread(mask_path,0)
    
    masked_img= cv2.bitwise_and(img,mask)
    
    # Calculatehistogram with maskand withoutmask
    # Checkthird argument for mask
    hist_full= cv2.calcHist([img],[0],None,[256],[0,256])
    hist_mask= cv2.calcHist([img],[0],mask,[256],[0,256])
    
    plt.subplot(221), 
    plt.imshow(img, 'gray')
    plt.subplot(222), 
    plt.imshow(mask,'gray')
    plt.subplot(223), 
    plt.imshow(masked_img, 'gray')
    plt.subplot(224), 
    plt.plot(hist_full), 
    plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.savefig(save_folder+'/histogram_obraz-maska.png')