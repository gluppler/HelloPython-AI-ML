#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:13:31 2025

@author: gluppler
"""

import numpy as np
import cv2

sE1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
sE1 = sE1.astype(np.uint8)

sE2 = np.array([ [1,1,1],[0,1,0],[0,1,0] ])
sE2 = sE2.astype(np.uint8)

sE3 = np.array([ [0,0,0],[0,0,0],[1,1,1],[0,1,0],[0,1,0] ])
sE3 = sE3.astype(np.uint8)