import torch

### concat_dim0
# torch.cat((a, b), dim=0) concatenates along rows (stack vertically).
# Shape rule (2D):
# if a is (m, n) and b is (k, n), then cat dim=0 -> (m + k, n)
a = torch.zeros((2, 4))
b = torch.ones((3, 4))

out = torch.cat((a, b), dim=0)
print("Example 1 — torch.cat dim=0 (vertical concat):")
print("Tensor a shape:", a.shape)
print(a)
print("\nTensor b shape:", b.shape)
print(b)
print("\nResult out = torch.cat((a, b), dim=0):")
print(out)
print("out.shape:")
print(out.shape)

### concat_dim1
# torch.cat((a, b), dim=1) concatenates along columns (stack horizontally).
# Shape rule (2D):
# if a is (m, n) and b is (m, k), then cat dim=1 -> (m, n + k)
a = torch.zeros((2, 4))
b = torch.ones((2, 4))

out = torch.cat((a, b), dim=1)
print("\nExample 2 — torch.cat dim=1 (horizontal concat):")
print("Tensor a shape:", a.shape)
print(a)
print("\nTensor b shape:", b.shape)
print(b)
print("\nResult out = torch.cat((a, b), dim=1):")
print(out)
print("out.shape:")
print(out.shape)

# =====================================================================
# Case 1: Stacking along dim=0 (New dimension is inserted at index 0)
# =====================================================================
# Initialize two 2x3 tensors
a = torch.linspace(1,6,6).view(2,3)
b = torch.linspace(7,12,6).view(2,3)

print("--- Original Tensors a and b ---")
print("Tensor a:\n", a) # Prints tensor 'a' with shape [2, 3]
print("Tensor b:\n", b) # Prints tensor 'b' with shape [2, 3]


# Formula: out[0, :, :] = a and out[1, :, :] = b
# Shape Transformation: [2, 3] + [2, 3] -> [2(new), 2, 3] -> torch.Size([2, 2, 3])
out = torch.stack((a,b), dim=0)

print("\n--- Stacking along dim=0 ---")
print("Stacked tensor (dim=0):\n", out) # Prints the 3D tensor containing block 'a' then block 'b'
print("Shape of output:", out.shape)     # Prints torch.Size([2, 2, 3])


# =====================================================================
# Case 2: Stacking along dim=1 (New dimension is inserted at index 1)
# =====================================================================
# Resetting tensors
a = torch.linspace(1,6,6).view(2,3)
b = torch.linspace(7,12,6).view(2,3)

# Formula: out[i, 0, :] = a[i, :] and out[i, 1, :] = b[i, :]
# Shape Transformation: [2, 3] + [2, 3] -> [2, 2(new), 3] -> torch.Size([2, 2, 3])
out = torch.stack((a,b), dim=1)

print("\n--- Stacking along dim=1 ---")
print("Stacked tensor (dim=1):\n", out) # Prints the 3D tensor where rows of 'a' and 'b' interleave
print("Shape of output:", out.shape)     # Prints torch.Size([2, 2, 3])