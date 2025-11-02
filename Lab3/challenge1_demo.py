#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:26:15 2025

@author: gluppler
"""

# challenge1_demo.py
import numpy as np
import cv2

# Single uint8 pixel
a = np.array([[160]], dtype=np.uint8)   # 1×1 image works everywhere

# NumPy addition (wraparound)
b = a + 100
print("numpy + :", int(b[0, 0]))  # expected 4

# OpenCV addition (saturating)
add_value = np.full_like(a, 100, dtype=np.uint8)
c = cv2.add(a, add_value)
print("cv2.add  :", int(c.flatten()[0]))  # expected 255
