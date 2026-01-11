#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:14:01 2025

@author: gluppler
"""
import numpy as np
import cv2
from matplotlib import pyplot as pt
from filter import lpfilter
cman = cv2.imread("/Users/gluppler/Downloads/School/test/Lab8_Test Images_Filter_Function/cameraman.bmp", 0)

F = np.fft.fft2(cman)

Fs = np.fft.fftshift(F)
[nrow,ncol] = cman.shape
print("Choose a type(Ideal - 1 / Butterworth - 2 / Gaussian - 3):")
types = input()
print("Enter a radius:")
radius = int(input())

if types == '1':
    H = lpfilter(nrow,ncol,"ideal",radius)
    print(f"Ideal with radius of {radius}")
elif types == '2':
    H = lpfilter(nrow,ncol,"btw",radius,5)
    print(f"Butterworth with radius of {radius}")
else:
    H = lpfilter(nrow,ncol,"gaussian",radius)
    print(f"Gausssian with radius of {radius}")

H = 1-H

G = H * Fs

Gs = np.fft.ifftshift(G)
g = np.fft.ifft2(Gs)

g = g.real

g[g < 0] = 0
g[g > 255] = 255

pt.figure(figsize=(12,8))
pt.subplot(1,3,1)
pt.imshow(H, cmap="gray")
pt.title("High Pass Filter")
pt.axis("off")
pt.subplot(1,3,2)
pt.imshow(np.log(1+abs(G)), cmap="gray")
pt.title("Filtered Fourier Coefficients")
pt.axis("off")
pt.subplot(1,3,3)
pt.imshow(g, cmap="gray")
pt.title("Filtered Image")
pt.axis("off")
pt.tight_layout()

pt.show()



