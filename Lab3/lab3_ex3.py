#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:44:11 2025

@author: gluppler
"""

import cv2
import numpy as np
from pathlib import Path
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab3_TestImages/")
INPUT = BASE / "ufo.bmp"
OUT_LINEAR = BASE / "baboon_linear.png"
OUT_HISTEQ = BASE / "baboon_histeq.png"

if not INPUT.exists():
    print(f"{INPUT} not found.")
    sys.exit(1)

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)

# Try linear contrast
alpha, beta = 2.0, 20
linear = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imwrite(str(OUT_LINEAR), linear)

# Try histogram equalization
histeq = cv2.equalizeHist(img)
cv2.imwrite(str(OUT_HISTEQ), histeq)

cv2.imshow("Original", img)
cv2.imshow("Linear", linear)
cv2.imshow("HistEq", histeq)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Saved linear and histeq results.")
