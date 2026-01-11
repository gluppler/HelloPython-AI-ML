#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:10:42 2025

@author: gluppler
"""

#!/usr/bin/env python3
# Challenge 1 (Lab 7)

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/gluppler/Downloads/School/test/Lab7_Test Images_Filter_Function")
from filter import lpfilter

img_path = "/Users/gluppler/Downloads/School/test/Lab7_Test Images_Filter_Function/cameraman.bmp"
out_dir = "/Users/gluppler/Downloads/School/test/Lab7/"

img = cv2.imread(img_path, 0)
if img is None:
    raise SystemExit("ERROR: cameraman.bmp not found.")

nrow, ncol = img.shape
cutoff = 25
order = 2

# Low-pass → High-pass
L = {
    "Ideal": lpfilter(nrow, ncol, "ideal", cutoff),
    f"Butterworth_n{order}": lpfilter(nrow, ncol, "btw", cutoff, order),
    "Gaussian": lpfilter(nrow, ncol, "gaussian", cutoff)
}
Hhp = {name + "_HP": 1 - H for name, H in L.items()}

F = np.fft.fft2(img)
Fs = np.fft.fftshift(F)

results = {}

for name, H in Hhp.items():
    G = H * Fs
    Gs = np.fft.ifftshift(G)
    g = np.fft.ifft2(Gs)

    g = np.real(g)
    g = g - g.min()
    if g.max() != 0:
        g = g / g.max() * 255

    g = np.uint8(g)
    results[name] = g

    save_path = out_dir + f"lab7_ch1_{name}.png"
    cv2.imwrite(save_path, g)
    print("Saved:", save_path)

# Visualize
plt.figure(figsize=(12,10))
i = 1
for name, H in Hhp.items():
    plt.subplot(3,3,i); plt.title(name + " H"); plt.imshow(H, cmap="gray"); plt.axis("off"); i += 1
    plt.subplot(3,3,i); plt.title(name + " Output"); plt.imshow(results[name], cmap="gray"); plt.axis("off"); i += 1

plt.tight_layout()
plt.show()
