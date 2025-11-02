import numpy as np

# --- Exercise 1 ---
arr_4x4 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
arr_5x4 = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
])
arr_4x1 = np.array([[1],[2],[3],[4]])

print("4x4 array:\n", arr_4x4)
print("\n5x4 array:\n", arr_5x4)
print("\n4x1 array:\n", arr_4x1)
print("\ndtypes:", arr_4x4.dtype, arr_5x4.dtype, arr_4x1.dtype)

# --- Challenge 1 (fixed) ---
# Create with a safe dtype then convert to uint8 so overflow/wrap is demonstrated.
temp = np.array([[255, 256, 257]])        # default (big enough) integer dtype
array1D_uint8 = temp.astype(np.uint8)     # conversion wraps modulo 256
print("\nChallenge 1 – uint8 overflow (wrap) demonstration:")
print("original:", temp)
print("as uint8:", array1D_uint8)
# Expected printed result: [[255 256 257]] and [[255   0   1]]

# --- Challenge 2 ---
extra_row = np.array([[2, 0, 1, 4]])
arr_5x4_concat = np.concatenate((arr_4x4[:2], extra_row, arr_4x4[2:]), axis=0)
print("\nChallenge 2 – concatenated 5x4 array:")
print(arr_5x4_concat)
