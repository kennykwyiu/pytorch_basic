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


# --- Check current shape ---
# Print the shape of `out` so we know what we’re transposing.
print("out.shape:")
print(out.shape)

# --- Transpose using torch.t (2D only) ---
# `torch.t(out)` transposes a 2D tensor by swapping dim 0 and dim 1.
# If out is shape (m, n), torch.t(out) is shape (n, m).
print("\ntorch.t(out) (2D transpose):")
print(torch.t(out))

# --- Transpose using torch.transpose (works for any tensor dims) ---
# `torch.transpose(out, 0, 1)` swaps dimension 0 and 1 (same result as torch.t for 2D tensors).
out_T = torch.transpose(out, 0, 1)

print("\ntorch.transpose(out, 0, 1) result:")
print(out_T)
print("out_T.shape:")
print(out_T.shape)

### transpose_3d_swap_dim0_dim1
print("transpose_3d_swap_dim0_dim1")

# Create a 3D tensor with shape (1, 2, 3):
# - dim=0 size 1
# - dim=1 size 2
# - dim=2 size 3
a = torch.rand(1, 2, 3)

print("Input tensor a:")
print(a)

print("\na.shape (expected torch.Size([1, 2, 3])):")
print(a.shape)

# Transpose by swapping dimension 0 and 1.
# For a shape (1, 2, 3), swapping dims 0 and 1 produces shape (2, 1, 3).
transpose = torch.transpose(a, 0, 1)

print("\ntranspose = torch.transpose(a, 0, 1):")
print(transpose)

print("\ntranspose.shape (expected torch.Size([2, 1, 3])):")
print(transpose.shape)