import cv2 
from cvzone.HandTrackingModule import HandDetector 
import numpy as np 
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=20
imgSize=300
counter=0

folder = "C:/Users/novan/OneDrive/Desktop/Sign language detection/Data/Thank you"



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure bbox coordinates are valid
        if w > 0 and h > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 225
            
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            
            # Check if imgCrop is not empty
            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                aspectratio = h / w
                
                if aspectratio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                
                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
