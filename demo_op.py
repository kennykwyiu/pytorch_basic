import torch

# Tensor addition demo

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print("Initial tensor a:")
print(a)
print("Initial tensor b:")
print(b)

print("\na + b:")
print(a + b)
print("\na.add(b) (out-of-place):")
print(a.add(b))
print("\na after out-of-place add (unchanged):")
print(a)
print("\ntorch.add(a, b):")
print(torch.add(a, b))
print("\na.add_(b) (in-place):")
print(a.add_(b))
print("\na after in-place add (updated):")
print(a)

# Tensor subtraction demo

print("\nCurrent tensor a before subtraction:")
print(a)
print("Current tensor b before subtraction:")
print(b)

print("\na - b:")
print(a - b)
print("\na.sub(b) (out-of-place):")
print(a.sub(b))
print("\na after out-of-place sub (unchanged):")
print(a)
print("\ntorch.sub(a, b):")
print(torch.sub(a, b))
print("\na.sub_(b) (in-place):")
print(a.sub_(b))
print("\na after in-place sub (updated):")
print(a)

# multiplication

print("\nCurrent tensor a before multiplication:")
print(a)
print("Current tensor b before multiplication:")
print(b)

print("\na * b:")
print(a * b)
print("\na.mul(b) (out-of-place):")
print(a.mul(b))
print("\na after out-of-place mul (unchanged):")
print(a)
print("\ntorch.mul(a, b):")
print(torch.mul(a, b))
print("\na.mul_(b) (in-place):")
print(a.mul_(b))
print("\na after in-place mul (updated):")
print(a)

# division

print("\nCurrent tensor a before division:")
print(a)
print("Current tensor b before division:")
print(b)

print("\na / b:")
print(a / b)
print("\na.div(b) (out-of-place):")
print(a.div(b))
print("\na after out-of-place div (unchanged):")
print(a)
print("\ntorch.div(a, b):")
print(torch.div(a, b))
print("\na.div_(b) (in-place):")
print(a.div_(b))
print("\na after in-place div (updated):")
print(a)

### matmul

# For matrix multiplication: (m x n) @ (n x p) -> (m x p)
# Here: (2 x 1) @ (1 x 2) -> (2 x 2)
a = torch.ones(2, 1)
b = torch.ones(1, 2)

# Shape check to make dimension matching explicit
print("\nShape of a:", a.shape)
print("Shape of b:", b.shape)
print("Shape of a @ b:", (a @ b).shape)

# Different APIs for 2D matrix multiplication (same result)
print("\na @ b:")
print(a @ b)
print("\na.matmul(b):")
print(a.matmul(b))
print("\ntorch.matmul(a, b):")
print(torch.matmul(a, b))
print("\ntorch.mm(a, b):")
print(torch.mm(a, b))
print("\na.mm(b):")
print(a.mm(b))

### tensor
# For tensors with rank > 2, matmul treats the last two dims as matrices
# and broadcasts leading dims if needed.
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)

# Last two dims: (3 x 4) @ (4 x 3) -> (3 x 3)
print("\nTensor matmul result:")
print(a.matmul(b))
print("\nTensor matmul result shape:")
print(a.matmul(b).shape)

# 建立 4D tensor
a = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])  # (1, 1, 2, 3)
b = torch.tensor([[[[7, 8], [9, 10], [11, 12]]]])  # (1, 1, 3, 2)

out = a @ b

print("Shape:", out.shape)
print("Data:\n", out)

# 3 位學生的資料：每個人有 4 個特徵（例如：讀書時數、出席率、睡眠時間、模擬考分數）。
# 神經網路權重：這台機器會將這 4 個特徵轉換成 2 個輸出結果（例如：及格機率、得獎機率）。

# 1. 設定設備：優先使用 Mac 的 GPU (MPS)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"📡 正在使用的設備: {dev}")

# 2. 準備輸入資料 (Inputs)
# 假設有 3 個學生，每人有 4 個指標
# 維度: (3, 4)
students_data = torch.tensor([
    [10.0, 0.9, 8.0, 85.0],  # 學生 A
    [2.0, 0.5, 4.0, 40.0],  # 學生 B
    [8.0, 0.8, 7.0, 70.0]  # 學生 C
], device=dev)

# 3. 準備權重 (Weights)
# 我們需要把 4 個特徵轉換成 2 個預測值
# 為了能相乘，權重的形狀必須是 (4, 2)
# 我們隨機產生一些權重
weights = torch.randn(4, 2, device=dev)

# 4. 執行 AI 推論 (矩陣乘法)
# (3, 4) @ (4, 2) -> (3, 2)
predictions = students_data @ weights

print("-" * 30)
print(f"📊 輸入形狀: {students_data.shape}")
print(f"⚙️ 權重形狀: {weights.shape}")
print(f"🚀 預測結果形狀: {predictions.shape}")
print("-" * 30)
print("🔍 預測結果 (RAW DATA):")
print(predictions)

##############
