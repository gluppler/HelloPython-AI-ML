#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:41:36 2025

@author: gluppler
"""

# lab4_ex1_color_challenge.py
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

BASE = Path("/Users/gluppler/Downloads/School/test/Lab4_TestImages/")
COLOUR_IMG = BASE / "pillsetc.png"  # change to your color image (e.g. baymax.png)

if not COLOUR_IMG.exists():
    raise SystemExit(f"Missing {COLOUR_IMG}")

img = cv2.imread(str(COLOUR_IMG), cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit("Failed to read color image")

# compute and normalize hist for each channel
channels = ("b","g","r")
plt.figure(figsize=(8,4))
for i, col in enumerate(channels):
    hist = cv2.calcHist([img],[i],None,[256],[0,256]).ravel()
    hist = hist / hist.sum()
    plt.plot(hist, label=col)
plt.xlim([0,256]); plt.title("Normalized Color Histogram"); plt.xlabel("Intensity"); plt.legend()
out = BASE / f"{COLOUR_IMG.stem}_color_norm_hist.png"
plt.savefig(str(out))
print("Saved", out)
plt.show()
