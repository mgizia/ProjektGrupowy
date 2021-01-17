# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:15:33 2021

@author: emilk
"""
import os
from keras.models import load_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import skimage.io as io
import skimage.transform as trans


import numpy as np
import cv2
from PIL import Image, ImageDraw 



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def testGenerator_1_img(test_path,target_size = (256,256),flag_multi_class = False,as_gray = False):
        img = io.imread(test_path,as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(save_path+"/mask.png",img)


#założenie że obraz 1:1
def Predict_Img(img_path,zapis_path):
    model = load_model('unet_membrane.hdf5')    
    #testGene = testGenerator("data/membrane/test")
    #results = model.predict_generator(testGene,1,verbose=1)
    #saveResult("data/membrane/test",results)
    img = cv2.imread(img_path)
    h, w, c = img.shape
    t = testGenerator_1_img(img_path)
    result = model.predict_generator(t,1,verbose=1)
    saveResult(zapis_path,result)
    print("Zapisano maskę predykcji")


#Predict_Img('test.png','')