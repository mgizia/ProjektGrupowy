# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:39:46 2021

@author: gmart
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogramRGB(img,name='histogram.jpg'):

    #ZAKRESY
    r1 = np.arange(0,64)
    r2 = np.arange(64,128)
    r3 = np.arange(128,192)
    r4 = np.arange(192,256)
    
     
    
    #podzial na rgb
    #B,G,R = cv2.split(img)
    #plt.subplot(1, 3, 1)
    #plt.imshow(R)
    #plt.subplot(1, 3, 2)
    #plt.imshow(G)
    #plt.subplot(1, 3, 3)
    #plt.imshow(B)
    
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    plt.subplot(1, 3, 1)
    plt.imshow(r)
    plt.subplot(1, 3, 2)
    plt.imshow(g)
    plt.subplot(1, 3, 3)
    plt.imshow(b)
    
    #Histogram
    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
    print('r')
    axs[0].hist(r)
    print('g')
    axs[1].hist(g)
    print('b')
    axs[2].hist(b)
    plt.savefig(name)
    
    #polaczenie r g i b w obraz
    #img_join = cv2.merge((B,G,R))

#s = 'kontur.png'
#img = cv2.imread(s)
#histogramRGB(img)