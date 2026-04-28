import torch

### global_statistics
# Compute statistics across all elements in the tensor.
a = torch.rand(2, 2)
print("Input tensor a (for global statistics):")
print(a)
print("\ntorch.mean(a) (mean of all elements):")
print(torch.mean(a))
print("\ntorch.sum(a) (sum of all elements):")
print(torch.sum(a))
print("\ntorch.prod(a) (product of all elements):")
print(torch.prod(a))
