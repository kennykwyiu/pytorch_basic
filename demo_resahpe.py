import torch

import torch

### reshape_2x3_to_3x2
print("reshape_2x3_to_3x2")

# Create a 2x3 tensor of random values sampled uniformly from [0, 1).
x = torch.rand(2, 3)

print("Original tensor x shape (expected torch.Size([2, 3])):")
print(x.shape)
print("Original tensor x values:")
print(x)

# Reshape `x` from (2, 3) to (3, 2).
# This is valid because the total number of elements stays the same:
# 2*3 = 6 elements, and 3*2 = 6 elements.
# Note: reshape reinterprets the same data in row-major order (it does not “shuffle” values).
out = x.reshape(3, 2)

print("\nReshaped tensor out shape (expected torch.Size([3, 2])):")
print(out.shape)
print("Reshaped tensor out values:")
print(out)

# --- Transpose ---
# Transpose `out` by swapping its two dimensions.
# If `out` is shape (3, 2), then `torch.t(out)` returns shape (2, 3).
out = torch.t(out)

print("\nTransposed tensor out shape (expected torch.Size([2, 3])):")
print(out.shape)
print("Transposed tensor out values:")
print(out)