# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:45:43 2020

@author: Guy
"""

import cv2
import numpy as np
import math
import pyautogui
import os

pyautogui.FAILSAFE = False

#check screen size
screenWidth,screenHeight = pyautogui.size();

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    path, dirs, files = next(os.walk("./dataset/four/"))
    file_count = len(files)
    
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

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    

    k = cv2.waitKey(10)
    if k == 27:
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "dataset/four/fourfinger{}.jpeg".format(file_count)
        cv2.imwrite(img_name, thresh1)
        print("{} written!".format(img_name))
        print(file_count)
    