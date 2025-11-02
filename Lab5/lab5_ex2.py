#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:50:16 2025

@author: gluppler
"""

"""
lab5_ex2.py
Apply mean filters (3x3,5x5,7x7) to 'char.bmp' and save outputs for comparison.
"""
import cv2
import numpy as np
from pathlib import Path
BASE = Path("/Users/gluppler/Downloads/School/test/Lab5_TestImages/")
INFILE = BASE / "char.bmp"

if not INFILE.exists():
    raise SystemExit(f"Missing {INFILE}")

img = cv2.imread(str(INFILE), cv2.IMREAD_GRAYSCALE)

sizes = [3,5,7]
for k in sizes:
    out = cv2.blur(img, (k,k), borderType=cv2.BORDER_DEFAULT)
    dest = BASE / f"char_mean_{k}x{k}.png"
    cv2.imwrite(str(dest), out)
    print("Saved:", dest)

# Optional quick display (uncomment if running locally)
# cv2.imshow("original", img)
# cv2.imshow("mean3", cv2.imread(str(BASE / "char_mean_3x3.png"),0))
# cv2.waitKey(0); cv2.destroyAllWindows()
