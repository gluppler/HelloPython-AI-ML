#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:49:33 2025

@author: gluppler
"""

"""
lab5_ch1.py
Create 3x3 image and implement manual zero padding and replication (no copyMakeBorder).
"""
import numpy as np
from pathlib import Path
import cv2
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")

img = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.uint8)

def zero_pad_manual(img, pad=1):
    h,w = img.shape
    out = np.zeros((h+2*pad, w+2*pad), dtype=img.dtype)
    out[pad:pad+h, pad:pad+w] = img
    return out

def replicate_pad_manual(img, pad=1):
    h,w = img.shape
    out = np.zeros((h+2*pad, w+2*pad), dtype=img.dtype)
    out[pad:pad+h, pad:pad+w] = img
    # top
    out[:pad, pad:pad+w] = img[0:1, :]
    # bottom
    out[pad+h:, pad:pad+w] = img[-1:, :]
    # left
    out[:, :pad] = out[:, pad:pad+1]
    # right
    out[:, pad+w:] = out[:, pad+w-1:pad+w]
    return out

OUT_ZERO = BASE / "lab5_ch1_zero_manual.png"
OUT_REPL = BASE / "lab5_ch1_repl_manual.png"

z = zero_pad_manual(img, pad=1)
r = replicate_pad_manual(img, pad=1)

cv2.imwrite(str(OUT_ZERO), z * 25)
cv2.imwrite(str(OUT_REPL), r * 25)

print("Saved manual padded images:")
print(OUT_ZERO)
print(OUT_REPL)
