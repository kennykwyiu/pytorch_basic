import torch

### comparison
a = torch.rand(2, 3)
b = torch.rand(2, 3)

print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)

print("\ntorch.eq(a, b) (element-wise equality):")
print(torch.eq(a, b))
print("\ntorch.equal(a, b) (all elements and shape must match):")
print(torch.equal(a, b))

print("\ntorch.ge(a, b) (a >= b):")
print(torch.ge(a, b))
print("\ntorch.gt(a, b) (a > b):")
print(torch.gt(a, b))
print("\ntorch.le(a, b) (a <= b):")
print(torch.le(a, b))
print("\ntorch.lt(a, b) (a < b):")
print(torch.lt(a, b))
print("\ntorch.ne(a, b) (a != b):")
print(torch.ne(a, b))
