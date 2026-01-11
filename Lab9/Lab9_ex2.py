#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:28:43 2025

@author: gluppler
"""
import numpy as np
import cv2

sE2 = np.array([[1,1,1], [0,1,0], [0,1,0]])
sE2 = sE2.astype(np.uint8)

sE3 = np.array([[0,0,0], [0,0,0], [1,1,1], [0,1,0], [0,1,0]])
sE3 = sE3.astype(np.uint8)



sp_image = np.zeros((5,5))
sp_image[2,2] = 1

rSE2 = np.rot90(sE2,2)
rSE3 = np.rot90(sE3,2)

dil2 = cv2.dilate(sp_image, rSE2, iterations=1)
dil3 = cv2.dilate(sp_image, rSE3, iterations=1)

print (dil2)
print (dil3)