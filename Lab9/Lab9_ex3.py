#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:33:51 2025

@author: gluppler
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

sE = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)

rSE = np.rot90(sE, 2)

img = cv2.imread(r"/Users/gluppler/Downloads/School/test/Lab9_TestImages/charBinary.bmp",0)
ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img = binary_image // 255

dil_img = cv2.dilate(img, rSE, iterations = 2)

plt.figure()

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(dil_img, cmap="gray")
plt.title("Repaired Image")
plt.axis("off")



