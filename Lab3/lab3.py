#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:12:11 2025

@author: gluppler
"""

import numpy as np
import cv2

img = cv2.imread(r"/Users/gluppler/Downloads/School/test/Lab3_TestImages/baymax.png", 0)
cv2.imshow("Baymax", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

