#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:35:05 2025

@author: gluppler
"""

import numpy as np
import cv2
from matplotlib import pyplot as pt
from filter import ntfilter

img = cv2.imread(r"/Users/gluppler/Downloads/School/test/Lab8_Test Images_Filter_Function/cameraman.bmp",0)
F = np.fft.fft2(img)
Fs = np.fft.fftshift(F)

[nrow,ncol] = img.shape

notch = ntfilter(nrow,ncol,"ideal",5,76,130)
notch = notch * ntfilter(nrow,ncol,"ideal",5,128,78)
notch = notch * ntfilter(nrow,ncol,"ideal",5,104,151)
notch = notch * ntfilter(nrow,ncol,"ideal",5,126,178)
notch = notch * ntfilter(nrow,ncol,"ideal",5,105,101)

notch_filtered = Fs * notch

Gs = np.fft.ifftshift(notch_filtered)
g = np.fft.ifft2(Gs)
g = g.real
g[g > 225] = 255

pt.figure()

pt.subplot(2,2,1)
pt.imshow(g, cmap="gray")
pt.title("Cameraman + Periodic Noise")
pt.axis("off")

pt.subplot(2,2,2)
pt.imshow(np.log(1+abs(Fs)),cmap="gray")
pt.title("Spectrum")
pt.axis("off")

pt.subplot(2,2,3)
pt.imshow(notch,cmap="gray")
pt.title("notch Filter")
pt.axis("off")

pt.subplot(2,2,4)
pt.imshow(np.log(1+abs(notch_filtered)),cmap="gray")
pt.title("Spectrum - After Filtering")
pt.axis("off")

pt.show()