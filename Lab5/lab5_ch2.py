#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:51:04 2025

@author: gluppler
"""

"""
lab5_ch2.py
Implement geometric mean filter (3x3) for 'char.bmp'.
Geometric mean = (product of pixels)^(1/N). We avoid zeros by adding small epsilon.
"""
import cv2
import numpy as np
from pathlib import Path
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")
INFILE = BASE / "char.bmp"
OUT = BASE / "char_geom_3x3.png"

if not INFILE.exists():
    raise SystemExit(f"Missing {INFILE}")

img = cv2.imread(str(INFILE), cv2.IMREAD_GRAYSCALE).astype(np.float64)

# padding reflect to keep borders
pad = 1
padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

h, w = img.shape
out = np.zeros_like(img)

eps = 1e-6
for i in range(h):
    for j in range(w):
        win = padded[i:i+3, j:j+3].astype(np.float64) + eps
        prod = np.prod(win)          # product of 9 values
        geom = prod ** (1.0 / 9.0)   # 9th root
        out[i,j] = geom

# clip and convert back to uint8
out_u8 = np.clip(out, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT), out_u8)
print("Saved geometric mean filtered image:", OUT)
