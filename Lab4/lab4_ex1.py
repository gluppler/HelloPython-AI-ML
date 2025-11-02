#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:38:54 2025

@author: gluppler
"""

# lab4_ex1.py
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab4_TestImages/")  # change if needed
FILES = ["book.bmp", "car.bmp", "building.bmp", "subway.bmp"]

for name in FILES:
    p = BASE / name
    if not p.exists():
        print(f"Missing {p} — please place it in {BASE}")
        continue

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read", p); continue

    hist = cv2.calcHist([img], [0], None, [256], [0,256]).ravel()
    n_pixels = img.size
    norm_hist = hist / n_pixels

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title(f"{name} - Histogram")
    plt.plot(hist)
    plt.xlim([0,256])
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.title(f"{name} - Normalized Histogram")
    plt.plot(norm_hist)
    plt.xlim([0,256])
    plt.xlabel("Intensity")
    plt.ylabel("Normalized count")

    out = BASE / f"{name}_histograms.png"
    plt.tight_layout()
    plt.savefig(str(out))
    print(f"Saved {out}")
    plt.show()
