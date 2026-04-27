import torch

### comparison
# Create two random tensors with the same shape for element-wise comparison.
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

### sorting
# Sort values along dim=0 (column-wise): each column is sorted top-to-bottom.
a = torch.tensor([[1, 4, 4, 3, 5],
                  [2, 3, 1, 3, 5]])
print("\nSorting demo tensor a:")
print(a)
print("\nShape of sorting tensor a:")
print(a.shape)
print("\ntorch.sort(a, dim=0, descending=False):")
print(torch.sort(a, dim=0, descending=False))

### topk

# Use topk/kthvalue to retrieve largest-k values and k-th smallest values.
a = torch.tensor([[2, 4, 3, 1, 5],
                  [2, 3, 5, 1, 4]])

print("\nTop-k demo tensor a:")
print(a)
print("\nShape of top-k tensor a:")
print(a.shape)

print("\ntorch.topk(a, k=2, dim=0) (top 2 values per column):")
print(torch.topk(a, k=2, dim=0))

print("\ntorch.kthvalue(a, k=2, dim=0) (2nd smallest per column):")
print(torch.kthvalue(a, k=2, dim=0))
print("\ntorch.kthvalue(a, k=2, dim=1) (2nd smallest per row):")
print(torch.kthvalue(a, k=2, dim=1))