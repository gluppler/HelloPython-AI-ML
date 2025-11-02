#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:52:25 2025

@author: gluppler
"""

"""
lab5_ch3.py
Implement midpoint filter (5x5) for 'char.bmp'.
Midpoint filter = (max + min) / 2 inside the window.
"""
import cv2
import numpy as np
from pathlib import Path
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")
INFILE = BASE / "char.bmp"
OUT = BASE / "char_midpoint_5x5.png"

if not INFILE.exists():
    raise SystemExit(f"Missing {INFILE}")

img = cv2.imread(str(INFILE), cv2.IMREAD_GRAYSCALE).astype(np.float64)
pad = 2
padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

h, w = img.shape
out = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        win = padded[i:i+5, j:j+5]
        mn = win.min()
        mx = win.max()
        out[i,j] = 0.5 * (mn + mx)

out_u8 = np.clip(out, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT), out_u8)
print("Saved midpoint filtered image:", OUT)
