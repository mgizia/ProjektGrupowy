# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 23:37:21 2021

@author: emilk
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def distance(p0, p1): 
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)         

def Sr_min_max(img_path,mask_path,save_path):
    img = cv2.imread(img_path,0)
    mask = cv2.imread(mask_path,0)
    
    #ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    cnt = contours[0]
    M = cv2.moments(cnt)
    #print (M)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img,mask = mask)
    print ("sr: "+str(cx)+" , "+str(cy))
    print ("min: "+str(min_val)+" max: "+str(max_val))
    print ("min_loc: "+str(min_loc)+" max_loc: "+str(max_loc))
    
    mean_val = cv2.mean(img,mask = mask)
    
    print("Sredni kolor: "+str(mean_val))
    
    #odległoci od srodka ciężkoci max oraz min wartosci 
    odl_max = distance((cx,cy), max_loc)
    odl_min = distance((cx,cy), min_loc)
    min_max_zroznicowanie_jasnosci = max_val - min_val
    print("odl_max: "+str(odl_max))
    print("odl_min: "+str(odl_min))
    print("min_max_zroznicowanie_jasnosci: "+str(min_max_zroznicowanie_jasnosci))
    
    img = cv2.imread(img_path)
    cv2.circle(img,(cx,cy),2,(0,0,255),2)
    cv2.circle(img,min_loc,2,(0,255,0),2)
    cv2.circle(img,max_loc,2,(255,0,0),2)
    plt.figure(1)
    plt.imshow(img)
    cv2.imwrite(save_path+"/sr_min_max.png",img)

    return min_val, max_val, min_loc, max_loc, mean_val, odl_max, odl_min, min_max_zroznicowanie_jasnosci
    
#s = 'roboczy/obraz_maska.png'
#s1 = 'roboczy/mask.png'
#s2 ='roboczy/kwantyzacja_6_kolor.png'
#Sr_min_max(s,s1,'roboczy')
