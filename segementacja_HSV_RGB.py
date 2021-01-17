# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:38:20 2021

@author: emilk
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
         
from PIL import Image, ImageDraw 
import math as m
import skimage.transform as trans

def RGB_segmentacja_kolru(path,save_path = 'roboczy/segmentacja_kolor.png'):

    Img1 = cv2.imread(path)    
    h, w, c = Img1.shape
    Img1= cv2.resize(Img1, (h,w), interpolation = cv2.INTER_AREA)
    plt.figure(1)
    plt.imshow(Img1)

    kernel = np.ones((3,3),np.uint8)
    lower  =  np.array([ 50 , 60 , 60 ], dtype=np.uint8) 
    upper  =  np.array([ 80 , 80 , 80 ], dtype=np.uint8)
    mask = cv2.inRange(Img1, lower, upper)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.GaussianBlur(mask,(3,3),100)
    cv2.imwrite(save_path,mask)
    print("Jasne smugi:")

    plt.figure(2)
    plt.imshow(mask)

def HSV_segmentacja_kolru(path,save_path = 'roboczy/segmentacja_kolor_hsv.png'):

    Img1 = cv2.imread(path)    
    h, w, c = Img1.shape
    Img1= cv2.resize(Img1, (h,w), interpolation = cv2.INTER_AREA)
    
    #HSV
    hsv = cv2.cvtColor(Img1, cv2.COLOR_BGR2HSV)
    plt.figure(1)
    plt.imshow(hsv)

    kernel = np.ones((3,3),np.uint8)     
    #lower = np.array([0,20,60], dtype=np.uint8)
    #upper = np.array([20,255,255], dtype=np.uint8)
 
    lower = np.array([0,0,60], dtype=np.uint8)
    upper = np.array([360,40,255], dtype=np.uint8)
    
    mask = cv2.inRange(Img1, lower, upper)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    #mask = cv2.GaussianBlur(mask,(3,3),100)
    cv2.imwrite(save_path,mask)
    print("Jasne smugi:")

    plt.figure(2)
    plt.imshow(mask)



#s1='roboczy/kwantyzacja_6_kolor.png'
#s = 'roboczy/obraz_maska.png'
#RGB_segmentacja_kolru(s1)
#HSV_segmentacja_kolru(s1,save_path = 'roboczy/hsv_segmentacja_kolor.png')

