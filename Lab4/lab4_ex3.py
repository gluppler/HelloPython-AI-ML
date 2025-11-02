#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:43:56 2025

@author: gluppler
"""

# lab4_ex3.py
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab4_TestImages/")
IMG = BASE / "coins.bmp"
OUT_MASK01 = BASE / "coins_mask_01.npy"       # saved as numpy 0/1 mask
OUT_MASK_VIS = BASE / "coins_mask_vis.png"     # 0/255 visualization
OUT_EXTRACT = BASE / "coins_extracted.png"

if not IMG.exists():
    raise SystemExit(f"Missing {IMG}")

coins = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)
if coins is None:
    raise SystemExit("Failed to read coins.bmp")

# choose threshold (from histogram choose ~100 — lab uses 100)
T = 100
mask01 = (coins >= T).astype(np.uint8)   # 0/1 mask
mask_vis = (mask01 * 255).astype(np.uint8)  # 0/255 for saving/view

# extraction: multiply original (uint8) by mask01 but first convert mask to same scale
# If mask01 is 0/1, multiplication works directly but we need uint8 -> keep values
extracted = coins * mask01  # coins (0..255) * 0/1 -> coins retained where mask==1

# save results
np.save(str(OUT_MASK01), mask01)
cv2.imwrite(str(OUT_MASK_VIS), mask_vis)
cv2.imwrite(str(OUT_EXTRACT), extracted)

print("Saved mask (0/1) ->", OUT_MASK01)
print("Saved mask visualization ->", OUT_MASK_VIS)
print("Saved extracted coins ->", OUT_EXTRACT)

# optional display
plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.imshow(coins, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask_vis, cmap="gray"); plt.title("Binary Mask (vis)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(extracted, cmap="gray"); plt.title("Extracted"); plt.axis("off")
plt.tight_layout(); plt.show()
