# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:01:04 2020

@author: emilk
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import math as m
import skimage.transform as trans


#im = cv2.imread('test.png')
#im1 = cv2.imread('test_kolory.png')
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#
#im2 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

def RysujKontur(im,imgray,path):
    ret,thresh = cv2.threshold(imgray,170,255,0) 
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    h, w, c = im.shape
    thresh = cv2.resize(thresh,(h,w))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    img = cv2.drawContours(im, contours, -1, (255,0,255), 3)
    
    cv2.imwrite(path+"/mask_thresh.png",thresh)
    cv2.imwrite(path+"/kontur.png",img)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return img, thresh 


def momenty(imgray):
    ret,thresh = cv2.threshold(imgray,127,255,0,cv2.THRESH_BINARY)
    plt.imshow(thresh)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
   # img.convertTo(thresh, CV_32SC1);
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    
#    asymetria(thresh)
    
    #kontur
    cnt = contours[0]
#    funkcja2(im2,cnt)
    #momenty
    M = cv2.moments(cnt)
    
    #srodek obszaru
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #pole
    Pole = cv2.contourArea(cnt)

    #dlugosc
    L = cv2.arcLength(cnt,True)

    #Solidność to stosunek powierzchni konturu do wypukłej powierzchni obszaru.
    #Solidity is the ratio of contour area to its convex hull area.
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(Pole)/hull_area
    
    #parametr =    długosc hull / dlugosc konturu   [0,1]
    hull_L = cv2.arcLength(hull,True)
    parametr = hull_L/L
    

    #Sprawdzenie wypukłosci
    w =  cv2.isContourConvex (cnt)
    if w : 
        wypuklosc = 1
    else :
        wypuklosc = 0
    
    #Współczynnik proporcji ( szerokosc / wysokosc )
    x,y,w,h  = cv2.boundingRect(cnt)
    szer_wys = float(w)/h
  
    #Wslolczynnik W9, cyrkularnosc ksztaltu
    W9 = (2*m.sqrt((m.pi)*Pole))/L
    
    #Zaokraglam do 6 miejsca po przecinku
    Pole = round(Pole,6)
    L = round(L,6)
    solidity = round(solidity,6)
    szer_wys = round(szer_wys,6)
    W9 = round(W9,6)

    parametr = round(parametr,6)
    
    print("Środek konturu : ( "+str(cx)+" , "+str(cy)+" )")
    print("Pole : "+str(Pole))
    print("Obwód : "+str(L))
    print("Solidity: " + str(solidity))
    print("Czy krzywa jest wypukła: "+str(wypuklosc))
    print("Szerokosc / wyskoszc = "+str(szer_wys)) 
    print("Współczynnik W9 :"+str(W9))
    
    #print("hull_L :"+str(hull_L))
    print("Parametr hull_L/L :"+str(parametr))
    
    return cx, cy, Pole, L, solidity, wypuklosc, szer_wys, W9, parametr


def funkcja1(img,img_gray):
    ret, thresh = cv2.threshold(img_gray, 127, 255,0)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = contours[0]

    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        #far = tuple(cnt[f][0])
        cv2.line(img,start,end,[0,255,0],3)
        #cv2.circle(img,far,5,[0,0,255],-1)
    cv2.imwrite("kontur_przyblizony.png",img)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()    
    
def funkcja2(imgray,cnt):
    
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)
    
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
    
    print(min_val)
    print(max_val)
    print(min_loc)
    print(max_loc)
    
    mean_val = cv2.mean(im1,mask = mask)
    print(mean_val)
    
    


def asymetria(thresh_square):
    G_X = cv2.reduce(thresh_square, 0 ,cv2.REDUCE_SUM)
    G_Y = cv2.reduce(thresh_square, 1 ,cv2.REDUCE_SUM, cv2.CV_32F)

    compare_val = cv2.compareHist(G_X ,G_Y ,cv2.HISTCMP_CORREL)
    print (compare_val)
 

#RysujKontur(im,imgray)
#momenty(imgray)
#funkcja1(im, imgray)