import cv2 as cv
from cvzone.HandTrackingModule import HandDetector as HD
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
import numpy as np
from time import time
import math

cap = cv.VideoCapture(0)
detect = HD(maxHands=1)

offset = 20
imgSize = 300

classifier = Classifier('Model/keras_model.h5', "Model/label.txt")
labels = [chr(ord("A")+i) for i in range(26)]
labels.remove("Q")
labels += ["Q"]

counter = 0
folder = "Data/Q/"
while True:
    
    success, img = cap.read()
    hands, img = detect.findHands(img)

    if hands:

        hand = hands[0]
        x, y, w, h = hand['bbox']

        base = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        #cv.imshow("Image White", base)
        #cv.imshow("Image Crop", imgCrop)

        asprat = h/w

        if asprat > 1:
            k = imgSize / h
            calc_w = math.ceil(k*w)
            imgRes = cv.resize(imgCrop, (calc_w, imgSize))
            
            wGap = math.ceil((imgSize-calc_w)/2)
            base[:, wGap:(calc_w+wGap)] = imgRes
            predictions, index = classifier.getPrediction(img)
            print(predictions, index)
        else:
            k = imgSize / w
            calc_h = math.ceil(k*h)
            imgRes = cv.resize(imgCrop, (imgSize, calc_h))
            
            hGap = math.ceil((imgSize-calc_h)/2)
            base[hGap:calc_h+hGap, :] = imgRes

        cv.imshow("Image on white background", base)

        
    cv.imshow("Image", img)
    
    if cv.waitKey(20) & 0xFF == ord("d"):
        break

cv.waitKey(0)