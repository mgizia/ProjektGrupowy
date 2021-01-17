# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:31:20 2020

@author: emilk
"""

import numpy as np
from skimage import io
from sklearn.cluster import KMeans
import cv2 

import matplotlib.pyplot as plt

def r_g_b_div(image):
    table_r=[[]]
    table_g=[[]]
    table_b=[[]]
    #podzielenie obrazu na wiersze
    for row in image:

        mid_table_r = []
        mid_table_g = []
        mid_table_b = []
        #podzielenie obrazu na pixele w ka¿dym wierszu
        for pixel in row:
           # print(len(row))
            r=pixel[0]
            g=pixel[1]
            b=pixel[2]

            mid_table_r.append(r)
            mid_table_g.append(g)
            mid_table_b.append(b)
        #dodanie kolejnego rzêdu pikseli    
        table_r.append(mid_table_r)
        table_g.append(mid_table_g)
        table_b.append(mid_table_b)
    table_r.pop(0)
    table_g.pop(0)
    table_b.pop(0)
    return np.array(table_r), np.array(table_g), np.array(table_b)

def r_g_b_join(r,g,b):
    image = [[[]]]
    for j in range(len(r)):
        mid_table = [[]]
        for i in range(len(r[0])):
            pixel = [r[j][i],g[j][i],b[j][i]]
            mid_table.append(pixel)
        mid_table.pop(0)
        image.append(mid_table)
        
    image.pop(0)
    return np.array(image)

def kwantyzacja(img,n_colors=20):
    X = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    quantized = centers[labels].reshape(img.shape).astype("uint8")
    return quantized

def kwantyzacjaNcolor(img,mask,save_path,k):
    #img = cv2.imread(img_path)
    #mask = cv2.imread(mask_path,0)
    
    plt.figure(1)
    plt.imshow(img)
    plt.figure(2)
    plt.imshow(mask)

    r,g,b = r_g_b_div(img)
    #img= img*mask
    r= cv2.bitwise_and(r,mask)
    g= cv2.bitwise_and(g,mask)
    b= cv2.bitwise_and(b,mask)
    obraz_maska = r_g_b_join(r,g,b)
    
    quant_r = kwantyzacja(r,k)
    quant_g = kwantyzacja(g,k)
    quant_b = kwantyzacja(b,k)
    
    quant = r_g_b_join(quant_r,quant_g,quant_b)
    plt.figure(3)
    plt.imshow(quant)
    
    cv2.imwrite(save_path+'/obraz_maska.png',obraz_maska)
    cv2.imwrite(save_path+'/kwantyzacja_'+str(k)+'_kolor.png',quant)

    return quant

