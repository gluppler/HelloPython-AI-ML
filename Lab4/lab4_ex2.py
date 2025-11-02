#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:43:16 2025

@author: gluppler
"""

# lab4_ex2.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab4_TestImages/")
IMG = BASE / "building.bmp"
OUT_EQ = BASE / "building_equalized.png"
OUT_FIG = BASE / "building_before_after_hist.png"

if not IMG.exists():
    raise SystemExit(f"Missing {IMG}")

img = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("Failed to read building.bmp")

img_eq = cv2.equalizeHist(img)
cv2.imwrite(str(OUT_EQ), img_eq)
print("Saved equalized image to", OUT_EQ)

# compute histograms
hist_orig = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
hist_eq   = cv2.calcHist([img_eq],[0],None,[256],[0,256]).ravel()

# plot
plt.figure(figsize=(12,5))

plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original"); plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(img_eq, cmap="gray")
plt.title("Equalized"); plt.axis("off")

plt.subplot(2,2,3)
plt.plot(hist_orig); plt.xlim([0,256]); plt.title("Hist - Original")
plt.subplot(2,2,4)
plt.plot(hist_eq); plt.xlim([0,256]); plt.title("Hist - Equalized")

plt.tight_layout()
plt.savefig(str(OUT_FIG))
print("Saved figure to", OUT_FIG)
plt.show()
