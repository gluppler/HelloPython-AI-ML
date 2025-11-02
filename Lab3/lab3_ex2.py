#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:39:08 2025

@author: gluppler
"""

import cv2
import numpy as np
from pathlib import Path
import sys

BASE = Path("/Users/gluppler/Downloads/School/test/Lab3_TestImages/")
inputs = [BASE / "word1.bmp", BASE / "word2.bmp", BASE / "word3.bmp"]
for fn in inputs:
    if not fn.exists():
        print(f"Missing {fn}")
        sys.exit(1)

imgs = [cv2.imread(str(fn), cv2.IMREAD_GRAYSCALE).astype(np.float64) for fn in inputs]

# Resize if necessary
min_h = min(im.shape[0] for im in imgs)
min_w = min(im.shape[1] for im in imgs)
imgs = [cv2.resize(im, (min_w, min_h)) for im in imgs]

acc = np.zeros_like(imgs[0], dtype=np.float64)
for im in imgs:
    acc += im

scaled = (acc / acc.max()) * 255.0
scaled = np.clip(scaled, 0, 255).astype(np.uint8)

OUT = BASE / "word_sum.png"
cv2.imshow("Summed", scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(str(OUT), scaled)
print(f"Saved {OUT}")
