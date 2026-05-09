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

print("--- Original Tensors a and b ---")
print("Tensor a:\n", a) # Prints tensor 'a' with shape [2, 3]
print("Tensor b:\n", b) # Prints tensor 'b' with shape [2, 3]

# Formula: out[i, 0, :] = a[i, :] and out[i, 1, :] = b[i, :]
# Shape Transformation: [2, 3] + [2, 3] -> [2, 2(new), 3] -> torch.Size([2, 2, 3])
out = torch.stack((a,b), dim=1)

print("\n--- Stacking along dim=1 ---")
print("Stacked tensor (dim=1):\n", out) # Prints the 3D tensor where rows of 'a' and 'b' interleave
print("Shape of output:", out.shape)     # Prints torch.Size([2, 2, 3])

# 初始化 a 和 b
a = torch.linspace(1,6,6).view(2,3)
b = torch.linspace(7,12,6).view(2,3)

print("--- Original Tensors a and b ---")
print("Tensor a:\n", a) # Prints tensor 'a' with shape [2, 3]
print("Tensor b:\n", b) # Prints tensor 'b' with shape [2, 3]

# =====================================================================
# Case 3: Stacking along dim=2 (New dimension is inserted at index 2)
# =====================================================================
# Formula: out[i, j, 0] = a[i, j] and out[i, j, 1] = b[i, j]
# Shape Transformation: [2, 3] + [2, 3] -> [2, 3, 2(new)] -> torch.Size([2, 3, 2])
out = torch.stack((a,b), dim=2)

print("\n--- Stacking along dim=2 ---")
print("Stacked tensor (dim=2):\n", out)
# Results in a 2x3 matrix where each element is a pair [a_val, b_val]
print("Shape of output:", out.shape)     # Prints torch.Size([2, 3, 2])

#每個最深處的元素」是指張量中最底層、無法再拆分的純量數字（Scalar）。
# 在 PyTorch 的多維陣列中，維度是層層包裹的：
# 第 0 層 (dim=0)：整個矩陣的橫列（Rows）。
# 第 1 層 (dim=1)：橫列裡面的直欄（Columns）。
# 最深處 (dim=2 / 最內層)：直欄裡面的具體數字（例如：1.0 或 7.0）。
#
# 🔍 用「剥洋蔥」來對比當你對 a 和 b 進行堆疊時：
# dim=0（最外層）：把整個 a 矩陣和整個 b 矩陣當成兩個大物件堆疊。
# dim=1（中間層）：進到矩陣內部，把 a 的整橫列和 b 的整橫列成對堆疊。
# dim=2（最深處）：直接一路進到最核心，把 a 的單個數字和 b 同位置的單個數字成對綁在一起。