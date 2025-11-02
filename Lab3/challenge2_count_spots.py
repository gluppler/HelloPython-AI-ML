#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 21:29:12 2025

@author: gluppler
"""

import cv2
import numpy as np
from pathlib import Path

# === Config ===
BASE = Path("/Users/gluppler/Downloads/School/test/Lab3_TestImages/")
A = BASE / "dice1.png"
B = BASE / "dice2.png"
THRESH = 50       # higher = less sensitive
MIN_AREA = 500    # ignore tiny spots

# === Load ===
a = cv2.imread(str(A))
b = cv2.imread(str(B))
ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
if ga.shape != gb.shape:
    gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))

# === Difference + threshold ===
diff = cv2.absdiff(ga, gb)
_, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# === Find regions ===
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]

# === Draw boxes ===
overlay = a.copy()
for i in valid:
    x, y, w, h, _ = stats[i]
    cx, cy = map(int, centroids[i])
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(overlay, str(i), (cx-10, cy+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# === Create visuals ===
# side-by-side comparison and heatmap
side_by_side = np.hstack([a, b])
heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

# === Save outputs ===
cv2.imwrite(str(BASE / "dice_diff_mask.png"), mask)
cv2.imwrite(str(BASE / "dice_diff_overlay.png"), overlay)
cv2.imwrite(str(BASE / "dice_diff_heatmap.png"), heatmap)
cv2.imwrite(str(BASE / "dice_side_by_side.png"), side_by_side)

print(f"Detected regions: {len(valid)}")
print("Saved:")
print(f" - Mask ............ dice_diff_mask.png")
print(f" - Overlay ......... dice_diff_overlay.png")
print(f" - Heatmap ......... dice_diff_heatmap.png")
print(f" - Side-by-side .... dice_side_by_side.png")

# === Optional display ===
cv2.imshow("Dice A vs B", side_by_side)
cv2.imshow("Difference Heatmap", heatmap)
cv2.imshow("Overlay (Labeled)", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

