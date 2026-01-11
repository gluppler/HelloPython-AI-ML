#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:06:05 2025

@author: gluppler
"""
import numpy as np
import cv2
from matplotlib import pyplot as pt

img = cv2.imread(r"/Users/gluppler/Downloads/School/test/Lab9_TestImages/coins.bmp",0)
cv2.imshow("Coins", img)
cv2.waitKey()

hist = cv2.calcHist([img], [0], None, [256], [0, 255])

pt.figure()
pt.xlabel("Bins")
pt.ylabel("Number of Pixels")

pt.plot(hist)
pt.xlim([0, 256])

pt.show()

[nrow, ncol] = img.shape
bi_img = np.zeros((nrow, ncol), dtype=np.uint8)

threshold = 100
for x in range(0, nrow):
    for y in range(0, ncol):
        if img[x,y] >= threshold:
            bi_img[x,y] = 0
        else:
            bi_img[x,y] = 1

sE = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
sE = sE.astype(np.uint8)

opening_output = cv2.morphologyEx(bi_img,cv2.MORPH_OPEN,sE)

pt.figure()

pt.subplot(1,3,1)
pt.imshow(img, cmap="gray")
pt.title("Original Image")
pt.axis("off")

pt.subplot(1,3,2)
pt.imshow(bi_img, cmap="gray")
pt.title("Binary Image")
pt.axis("off")

pt.subplot(1,3,3)
pt.imshow(opening_output, cmap="gray")
pt.title("Repaired Image")
pt.axis("off")

pt.show()