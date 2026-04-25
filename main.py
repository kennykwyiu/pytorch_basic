import sys

import torch

print(f"目前使用的 Python 路徑: {sys.executable}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"M4 GPU 加速是否可用: {torch.backends.mps.is_available()}")

a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

a = torch.Tensor(2, 3)
print(a)
print(a.type())

a = torch.ones(2, 2)
print(a)
print(a.type())

a = torch.eye(2, 2)
print(a)
print(a.type())

a = torch.zeros(2, 2)
print(a)
print(a.type())

b = torch.Tensor(2, 3)
b = torch.zeros_like(b)
b = torch.ones_like(b)
print(b)
print(b.type())

a = torch.rand(2, 2)
print(a)
print(a.type())

a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())

a = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(a)
print(a.type())

a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())

a = torch.arange(0, 11, 3)
print(a)
print(a.type())

a = torch.linspace(2, 10, 20)
print(a)
print(a.type())

a = torch.randperm(10)
print(a)
print(a.type())

###############
import numpy as np

a = np.array([[1, 2], [2, 3]])
print(a)
