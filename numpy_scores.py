import numpy as np

# -------------------- Task 1 --------------------
np.random.seed(42)
scores = np.random.randint(50, 101, (5, 4))

print("Scores:\n", scores)

# 3rd student in 2nd subject
print("\n3rd student, 2nd subject:", scores[2, 1])

# Last 2 students (all subjects)
print("\nLast 2 students:\n", scores[-2:, :])

# First 3 students in subjects 2 and 3
print("\nFirst 3 students, subjects 2 & 3:\n", scores[:3, 1:3])


# -------------------- Task 2 --------------------

# Column-wise mean (rounded to 2 decimals)
col_mean = np.round(scores.mean(axis=0), 2)
print("\nColumn-wise mean:", col_mean)

# Add curve using broadcasting
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve

# Ensure no value exceeds 100
curved_scores = np.clip(curved_scores, None, 100)

print("\nCurved Scores:\n", curved_scores)

# Row-wise max
row_max = curved_scores.max(axis=1)
print("\nRow-wise max:", row_max)


# -------------------- Task 3 --------------------

# Min-max normalization per row
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)

normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nNormalized Scores:\n", normalized)

# Index of single highest value
max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print("\nHighest value at (row, col):", max_index)

# All scores strictly above 90
above_90 = curved_scores[curved_scores > 90]
print("\nScores above 90:", above_90)
