#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:34:18 2025

@author: gluppler
"""

# challenge3_mask.py
import cv2
import numpy as np
from pathlib import Path

BASE = Path("/Users/gluppler/Downloads/School/test/Lab3_TestImages/")
IMG = BASE / "baboon.bmp"
OUT = BASE / "baboon_masked.png"

img = cv2.imread(str(IMG))
if img is None:
    raise SystemExit(f"Missing {IMG}")

h, w = img.shape[:2]

# circular mask in center
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite(str(BASE / "mask_preview.png"), mask)
cv2.imwrite(str(OUT), masked)

cv2.imshow("Mask", mask)
cv2.imshow("Masked", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Saved masked image to {OUT}")

