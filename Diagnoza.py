# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 00:25:21 2021

@author: emilk
"""


import numpy as np
import cv2
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt

from predict import Predict_Img
from konturyFunkcjeParametry import *
from kwantyzacja_k_srednich import *
from RGB import *
from zabawy_histogram_2 import RysujHistogram
from segementacja_HSV_RGB import HSV_segmentacja_kolru
from sr_min_max import Sr_min_max
import tensorflow as tf

def Diagnoza(sciezka_obraz, sciezka_folder='roboczy'):
    Predict_Img(sciezka_obraz,sciezka_folder)
    
    obraz = cv2.imread(sciezka_obraz)
    #plt.figure(1)
    #plt.imshow(obraz)
    maska = cv2.imread(sciezka_folder+'/mask.png')
    #plt.figure(2)
    #plt.imshow(maska)
    
    img = obraz
    #Liczę parametry zmiany na podstawie paski
    cx, cy, Pole, L, solidity, wypuklosc, szer_wys, W9, parametr = momenty(maska);
    
    #rysuję kontur na oryginalnym obrazie
    #img , thresh = RysujKontur(img,maska,sciezka_folder)
    #cv2.imwrite(sciezka_folder+"/kontur.png",img)

    #kwantyzacja k srednich
    #obraz = cv2.imread(sciezka_obraz)
    #obraz_Nkolor = kwantyzacjaNcolor(obraz,thresh,sciezka_folder,6)
    #obraz_3_kolor = kwantyzacjaNcolor(obraz,thresh,sciezka_folder,3)
    
    #Rysuje histogramy obrazu oryginalnego 
    #histogramRGB(obraz,'roboczy/histogram_obraz.jpg')
    
    #Rysuje histogramy obrazu kwantowanego 
    #histogramRGB(obraz_Nkolor,'roboczy/histogram_k6_kolor.jpg')
    #histogramRGB(obraz_3_kolor,'roboczy/histogram_k3_kolor.jpg')
    
    #Obliczam punkty Srodka ciężkoci minimum maximum oraz sredni kolor
    #Sr_min_max(sciezka_obraz,sciezka_folder+'/mask.png',sciezka_folder)

    #rysuje histogram zmiany bez tła
    #RysujHistogram(sciezka_obraz,sciezka_folder+'/mask_thresh.png',sciezka_folder)
    
    #Maska segmentacji z zarkesem hsv (próbuję kolory jasne ,szrosci wyciąć, potencjalne smugi itp)
    #HSV_segmentacja_kolru(sciezka_obraz)
    
    #Stworzyć wektor danych do predykcji wyniku
    #Predykcja wyniku
    #wczytanie modelu
    model = tf.keras.models.load_model('binarymelanoma_v3.h5')
    test = ([[Pole, L, solidity, wypuklosc, szer_wys, W9, parametr]])
    test = np.asarray(test).astype(np.float32)
    #predykcja
    #print(test)
    Pr = model.predict(test)
    #print(Pr) # they are outputs of the sigmoid, interpreted as probabilities p(y = 1 | x)
    #import numpy as np
    Pr = np.round(Pr).flatten()
    #print(Pr)
    #print(Pr.type)
    wynik = int(Pr[0])
    #1=czerniak
    #0=nie czerniak
    #wynik to int
    return wynik
  
#Diagnoza('test1.png')


