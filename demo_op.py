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
a = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]]) # (1, 1, 2, 3)
b = torch.tensor([[[[7, 8], [9, 10], [11, 12]]]]) # (1, 1, 3, 2)

out = a @ b

print("Shape:", out.shape)
print("Data:\n", out)
