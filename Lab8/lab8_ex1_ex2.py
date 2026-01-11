#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:15:22 2025

@author: gluppler
"""

import numpy as np
import cv2
from matplotlib import pyplot as pt
from filter import brfilter

img = cv2.imread(r"/Users/gluppler/Downloads/School/test/Lab8_Test Images_Filter_Function/cameraman.bmp",0)
F = np.fft.fft2(img)
Fs = np.fft.fftshift(F)

pt.imshow(np.log(1+abs(Fs)), cmap="gray")
pt.title("Click to see coordinates")
pt.axis("on")
pt.colorbar()
pt.show()

def calcLength(startPt, endPt):
    legX = endPt[0] - startPt[0]
    legY = endPt[1] - startPt[1]
    
    hypotenuse = np.sqrt(np.power(legX,2) + np.power(legY,2))
    return hypotenuse

[nrow,ncol] = img.shape
L = calcLength([126.8, 125.8], [127.5, 76.7])
I = brfilter(nrow,ncol,"ideal",L,10)
F_filtered = Fs * I

Gs = np.fft.ifftshift(F_filtered)
g = np.fft.ifft2(Gs)
g = g.real
g[g < 0] = 0
g[g > 255] = 255

pt.figure()

pt.subplot(2,2,1)
pt.imshow(g,cmap="gray")
pt.title("Cameraman + Periodic Noise")
pt.axis("off")

pt.subplot(2,2,2)
pt.imshow(np.log(1+abs(Fs)),cmap="gray")
pt.title("Spectrum")
pt.axis("off")

pt.subplot(2,2,3)
pt.imshow(I,cmap="gray")
pt.title("Band Reject Filter")
pt.axis("off")

pt.subplot(2,2,4)
pt.imshow(np.log(1+abs(F_filtered)),cmap="gray")
pt.title("Spectrum - After Filtering")
pt.axis("off")

pt.show()