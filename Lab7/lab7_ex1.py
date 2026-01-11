#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:06:40 2025

@author: gluppler
"""

#!/usr/bin/env python3
# Exercise 1 (Lab 7) - Using absolute paths

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/gluppler/Downloads/School/test/Lab7_Test Images_Filter_Function")
from filter import lpfilter

# INPUTS
img_path = "/Users/gluppler/Downloads/School/test/Lab7_Test Images_Filter_Function/cameraman.bmp"

# OUTPUT DIR
out_dir = "/Users/gluppler/Downloads/School/test/Lab7/"

# Load grayscale image
img = cv2.imread(img_path, 0)
if img is None:
    raise SystemExit("ERROR: cameraman.bmp not found at " + img_path)

# Forward FFT
F = np.fft.fft2(img)
Fs = np.fft.fftshift(F)
spectrum_shifted = np.log(1 + np.abs(Fs))

# Filter
nrow, ncol = img.shape
H = lpfilter(nrow, ncol, "ideal", 75)

# Multiply in frequency domain
G = H * Fs
filtered_spectrum = np.log(1 + np.abs(G))

# Inverse FFT
Gs = np.fft.ifftshift(G)
g = np.fft.ifft2(Gs)
g = np.real(g)
g[g < 0] = 0
g[g > 255] = 255
g_uint8 = np.uint8(g)

# Save output
save_path = out_dir + "lab7_ex1_filtered.png"
cv2.imwrite(save_path, g_uint8)
print("Saved:", save_path)

# Display for checking
plt.figure(figsize=(10,10))
plt.subplot(2,2,1); plt.title("Shifted Spectrum"); plt.imshow(spectrum_shifted, cmap='gray'); plt.axis("off")
plt.subplot(2,2,2); plt.title("Filter H"); plt.imshow(H, cmap='gray'); plt.axis("off")
plt.subplot(2,2,3); plt.title("Filtered Spectrum"); plt.imshow(filtered_spectrum, cmap='gray'); plt.axis("off")
plt.subplot(2,2,4); plt.title("Filtered Image"); plt.imshow(g_uint8, cmap='gray'); plt.axis("off")
plt.tight_layout()
plt.show()
