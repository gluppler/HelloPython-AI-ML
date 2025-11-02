import numpy as np

# --- Exercise 3 ---
# Create a 200x200 array filled with value 100
arr200 = np.full((200, 200), 100, dtype=np.uint8)
print("200x200 array filled with 100, shape:", arr200.shape)
print("Top-left 3x3 slice:\n", arr200[:3, :3])

# --- Challenge 3 ---
# Create the 7x7 patterned array using zeros + for-loops
arr7 = np.zeros((7, 7), dtype=np.int32)
val = 1
for x in range(0, 7, 2):        # rows 0,2,4,6
    for y in range(0, 7, 2):    # columns 0,2,4,6
        arr7[x, y] = val
        val += 1

print("\nChallenge 3 – 7x7 pattern:\n", arr7)
