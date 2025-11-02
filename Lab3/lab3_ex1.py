#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:04:02 2025

@author: gluppler
"""

import cv2
import numpy as np
from pathlib import Path
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab3_TestImages/")
INPUT = BASE / "cameraman.bmp"
OUT = BASE / "cameraman_tl_black.png"

if not INPUT.exists():
    print(f"ERROR: {INPUT} not found.")
    sys.exit(1)

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img[0:h//2, 0:w//2] = 0

cv2.imshow("Top-left blacked", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(str(OUT), img)
print(f"Saved output to {OUT}")
