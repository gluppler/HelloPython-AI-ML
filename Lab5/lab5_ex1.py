#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:46:49 2025

@author: gluppler
"""

"""
lab5_ex1.py
Create a 3x3 sample image and pad it using zero padding and replication via cv2.copyMakeBorder.
"""
import cv2
import numpy as np
from pathlib import Path
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")

# Create a simple 3x3 image of ints (example)
img = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]], dtype=np.uint8)

# Paths (for saving visualizations)
OUT_ZERO = BASE / "lab5_ex1_zeropad.png"
OUT_REPL = BASE / "lab5_ex1_replicate.png"
OUT_REFLECT = BASE / "lab5_ex1_reflect.png"

# Zero padding (one pixel each side)
zero = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
rep  = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_REPLICATE)
ref  = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_REFLECT_101)

# Save as images with scaling for visibility (multiply so numbers are visible)
cv2.imwrite(str(OUT_ZERO), zero * 25)
cv2.imwrite(str(OUT_REPL),  rep * 25)
cv2.imwrite(str(OUT_REFLECT),ref * 25)

print("Saved zero, replicate, reflect padded images to:")
print(OUT_ZERO)
print(OUT_REPL)
print(OUT_REFLECT)
