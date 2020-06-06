# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:50:16 2020

@author: Guy
"""

from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np
import math
import time
import pyautogui


pyautogui.FAILSAFE = False
SCREEN_X, SCREEN_Y = pyautogui.size()
CLICK = CLICK_MESSAGE = MOVEMENT_START = None


cap = cv2.VideoCapture(0)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

while(cap.isOpened()):
    # read image
    ret, img = cap.read()

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (200,70), (550,400), (0,255,0),0)
    crop_img = img[70:400, 200:550]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # show thresholded image
    cv2.imshow('Thresholded', thresh1)
    
    img_data = np.array(thresh1)
    img_data = img_data.astype('float32')
    img_data /= 255
    
    img_data= np.expand_dims(img_data, axis=0)
    img_data= np.expand_dims(img_data, axis=0)
    print (img_data.shape)
    
    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    
    result = model.predict_classes(img_data)
    resX,resY = pyautogui.size()
    x,y = pyautogui.position()
    # define actions required
    if result == 0:
        cv2.putText(img,"none", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        
    elif result == 1:
        cv2.putText(img, "up", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        if(y<resY):
            pyautogui.moveTo(x,y-5)
    elif result == 2:
        cv2.putText(img,"right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        if(x<resX):
            pyautogui.moveTo(x+5,y)
    elif result == 3:
        cv2.putText(img,"down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        if(y>0):
            pyautogui.moveTo(x,y+5)
    elif result == 4:
        cv2.putText(img, "left", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        if(x>0):
            pyautogui.moveTo(x-5,y)
    elif result == 5:
        cv2.putText(img, "click", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        pyautogui.click()
    else:
        cv2.putText(img,"none", (50, 50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    # all_img = np.hstack((drawing, crop_img))
    # cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break