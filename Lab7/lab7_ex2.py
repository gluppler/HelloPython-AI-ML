#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:09:51 2025

@author: gluppler
"""

#!/usr/bin/env python3
# Exercise 2 (Lab 7)

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

filters = {
    "Ideal": lpfilter(nrow, ncol, "ideal", cutoff),
    f"Butterworth_n{order}": lpfilter(nrow, ncol, "btw", cutoff, order),
    "Gaussian": lpfilter(nrow, ncol, "gaussian", cutoff)
}

F = np.fft.fft2(img)
Fs = np.fft.fftshift(F)

results = {}
spectra = {}

for name, H in filters.items():
    G = H * Fs
    spectra[name] = np.log(1 + np.abs(G))

    Gs = np.fft.ifftshift(G)
    g = np.fft.ifft2(Gs)
    g = np.real(g)
    g[g < 0] = 0
    g[g > 255] = 255
    results[name] = np.uint8(g)

    save_path = out_dir + f"lab7_ex2_{name}.png"
    cv2.imwrite(save_path, results[name])
    print("Saved:", save_path)

# Visualization 3×3 layout
plt.figure(figsize=(12,10))
i = 1
for name, H in filters.items():
    plt.subplot(3,3,i); plt.title(name + " H"); plt.imshow(H, cmap="gray"); plt.axis("off"); i += 1
    plt.subplot(3,3,i); plt.title(name + " Spectrum"); plt.imshow(spectra[name], cmap="gray"); plt.axis("off"); i += 1
    plt.subplot(3,3,i); plt.title(name + " Output"); plt.imshow(results[name], cmap="gray"); plt.axis("off"); i += 1

plt.tight_layout()
plt.show()
