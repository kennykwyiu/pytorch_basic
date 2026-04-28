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

### dimension_wise_statistics
# Compute statistics column-wise (dim=0).
a = torch.rand(2, 2)
print("\nInput tensor a (for dim=0 statistics):")
print(a)
print("\ntorch.mean(a, dim=0) (mean per column):")
print(torch.mean(a, dim=0))
print("\ntorch.sum(a, dim=0) (sum per column):")
print(torch.sum(a, dim=0))
print("\ntorch.prod(a, dim=0) (product per column):")
print(torch.prod(a, dim=0))

### index_of_extreme_values
# Return indices of max/min values per column.
print("\ntorch.argmax(a, dim=0) (index of max per column):")
print(torch.argmax(a, dim=0))
print("\ntorch.argmin(a, dim=0) (index of min per column):")
print(torch.argmin(a, dim=0))

### distribution_statistics
# Measure spread and central tendency of tensor values.
print("\ntorch.std(a) (standard deviation):")
print(torch.std(a))
print("\ntorch.var(a) (variance):")
print(torch.var(a))
print("\ntorch.median(a) (median value):")
print(torch.median(a))
print("\ntorch.mode(a) (most frequent value and index):")
print(torch.mode(a))