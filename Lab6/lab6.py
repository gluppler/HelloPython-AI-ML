#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:02:41 2025

@author: gluppler
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# --- User-adjustable path ---
IMAGE_DIR = r"/Users/gluppler/Downloads/School/test/Lab6_TestImages"
CAMERAMAN_FILENAME = "cameraman.bmp"

# --- Output directory ---
OUT_DIR = "lab6_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Exercise A: 2x2 DFT
# -------------------------
arr = np.array([[9, 8],
                [19, 60]], dtype=np.float64)

dft_arr = np.fft.fft2(arr)
rec_arr = np.fft.ifft2(dft_arr)
rec_arr = np.real_if_close(rec_arr)

print("Original array:\n", arr)
print("DFT coefficients:\n", dft_arr)
print("Reconstructed array (ifft2):\n", rec_arr)

# -------------------------
# Exercise B: simple image
# -------------------------
simple_image = np.zeros((256, 256), dtype=np.float64)
simple_image[124:132, 124:132] = 255.0

F = np.fft.fft2(simple_image)
F_shift = np.fft.fftshift(F)
spectrum = np.abs(F_shift)
spectrum_log = np.log1p(spectrum)

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1); plt.imshow(simple_image, cmap="gray"); plt.title("Simple Image"); plt.axis("off")
plt.subplot(2, 2, 2); plt.imshow(spectrum, cmap="gray"); plt.title("Spectrum"); plt.axis("off")
plt.subplot(2, 2, 3); plt.imshow(spectrum_log, cmap="gray"); plt.title("Spectrum Log"); plt.axis("off")

F_unshift = np.fft.ifftshift(F_shift)
rec_simple = np.abs(np.fft.ifft2(F_unshift))

plt.subplot(2, 2, 4); plt.imshow(rec_simple, cmap="gray"); plt.title("Reconstructed"); plt.axis("off")
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "exercise_B.png"), dpi=300)
plt.close()

# -------------------------
# Exercise C: cameraman image
# -------------------------
cam_path = os.path.join(IMAGE_DIR, CAMERAMAN_FILENAME)
if not os.path.exists(cam_path):
    raise FileNotFoundError(f"Expected image at {cam_path}")

img = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)
magnitude = np.abs(F_shift)
magnitude_log = np.log1p(magnitude)
phase = np.angle(F_shift)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(2, 2, 2); plt.imshow(magnitude, cmap="gray"); plt.title("Magnitude"); plt.axis("off")
plt.subplot(2, 2, 3); plt.imshow(magnitude_log, cmap="gray"); plt.title("Magnitude Log"); plt.axis("off")
plt.subplot(2, 2, 4); plt.imshow(np.abs(phase), cmap="gray"); plt.title("Phase"); plt.axis("off")
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "exercise_C.png"), dpi=300)
plt.close()

print(f"All images saved into: {OUT_DIR}")

