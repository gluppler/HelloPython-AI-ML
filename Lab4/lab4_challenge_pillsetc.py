#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:44:59 2025

@author: gluppler
"""

# lab4_challenge_pillsetc.py
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab4_TestImages/")
IMG = BASE / "pillsetc.png"
OUT_MASK = BASE / "pillsetc_mask.png"
OUT_SEG = BASE / "pillsetc_segmented.png"

if not IMG.exists():
    raise SystemExit(f"Missing {IMG}")

img_bgr = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise SystemExit("Failed to read pillsetc.png")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Otsu threshold to auto choose T (works well for many foreground/background)
_, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optional morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Keep original colours for foreground: use mask as 0/255
masked_color = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

# Save (convert BGR->RGB for matplotlib display if needed)
cv2.imwrite(str(OUT_MASK), mask)
cv2.imwrite(str(OUT_SEG), masked_color)
print("Saved mask ->", OUT_MASK)
print("Saved segmented color image ->", OUT_SEG)

# Show results
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB)); plt.title("Segmented (colors kept)"); plt.axis("off")
plt.tight_layout(); plt.show()
