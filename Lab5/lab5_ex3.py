#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:51:43 2025

@author: gluppler
"""

"""
lab5_ex3.py
Apply median filters (3x3,5x5,7x7) to 'char.bmp'.
"""
import cv2
from pathlib import Path
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")
INFILE = BASE / "char.bmp"

if not INFILE.exists():
    raise SystemExit(f"Missing {INFILE}")

img = cv2.imread(str(INFILE), cv2.IMREAD_GRAYSCALE)
for k in [3,5,7]:
    out = cv2.medianBlur(img, k)
    dest = BASE / f"char_median_{k}x{k}.png"
    cv2.imwrite(str(dest), out)
    print("Saved:", dest)
