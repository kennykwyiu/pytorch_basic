import torch

### global_statistics
# Compute statistics across all elements in the tensor.
# mean = average, sum = total, prod = multiply all values together.
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
# dim=0 means "down the rows", so result length equals number of columns.
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
# std/var show how spread out values are; median is the middle value.
print("\ntorch.std(a) (standard deviation):")
print(torch.std(a))
print("\ntorch.var(a) (variance):")
print(torch.var(a))
print("\ntorch.median(a) (median value):")
print(torch.median(a))
print("\ntorch.mode(a) (most frequent value and index):")
print(torch.mode(a))

### histogram_statistics
# histc counts how many values fall into each bin.
# bins=6 means split the value range into 6 intervals.
# min=0 and max=0 tells PyTorch to auto-use the tensor's min/max as range.
# If you want a fixed range for learning, use: torch.histc(a, bins=6, min=0, max=10)
a = torch.rand(2, 2) * 10
print("\nInput tensor a (scaled to 0~10) for histogram:")
print(a)
print("\ntorch.histc(a, bins=6, min=0, max=0):")
print(torch.histc(a, 6, 0, 0))
