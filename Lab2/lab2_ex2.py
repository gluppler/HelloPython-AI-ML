import numpy as np

# Create the base 4x4 array
array2D_int = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Original array:\n", array2D_int)

# --- Exercise 2 ---
# Change all elements in the first column to 2014
array2D_int[:, 0] = 2014

print("\nModified array (first column = 2014):\n", array2D_int)
