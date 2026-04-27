import torch

y = torch.rand(2, 2)
y = y * 10

print("Original tensor y (scaled by 10):")
print(y)
print("\ntorch.floor(y):")
print(torch.floor(y))
print("\ntorch.ceil(y):")
print(torch.ceil(y))
print("\ntorch.round(y):")
print(torch.round(y))
print("\ntorch.trunc(y):")
print(torch.trunc(y))
print("\ntorch.frac(y):")
print(torch.frac(y))
print("\ny % 2:")
print(y % 2)

print("\ny.floor() (method version):")
print(y.floor())