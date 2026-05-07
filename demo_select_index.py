import torch

# torch.where(condition, x, y) works element-wise (broadcasting rules apply):
# out[...]= x[...] when condition[...] is True
# out[...]= y[...] when condition[...] is False
x = torch.tensor([1., 2., 3.])
y = torch.tensor([10., 20., 30.])
cond = torch.tensor([True, False, True])
out = torch.where(cond, x, y)
print("Example 1 — 1D where:")
print("x:", x)
print("y:", y)
print("cond:", cond)
print("torch.where(cond, x, y):")
print(out)  # tensor([ 1., 20.,  3.])

a = torch.rand(4, 4)
b = torch.rand(4, 4)

print("\nExample 2 — 2D where with condition a > 0.5:")
print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)

out = torch.where(a > 0.5, a, b)
print("\nCondition (a > 0.5) — True means pick a, False means pick b:")
print(a > 0.5)
print("\nResult out = torch.where(a > 0.5, a, b):")
print(out)

### index_select_examples
# torch.index_select(input, dim, index) selects slices along dimension `dim`.
# Formula (conceptual for selecting along dim):
# out[i] = input.select(dim, index[i])  -> chosen along that dim.

a = torch.tensor([[10, 11, 12],
                  [20, 21, 22],
                  [30, 31, 32]])  # (3, 3)
idx = torch.tensor([2, 0])
print("\nExample 3 — torch.index_select along dim=0:")
print("Input a:")
print(a)
print("index idx:", idx)
out = torch.index_select(a, dim=0, index=idx)
print("Result out (selected rows in order idx):")
print(out)
# tensor([[30, 31, 32],
#         [10, 11, 12]])

print("\nExample 4 — torch.index_select along dim=0 with another index:")
a = torch.rand(4, 4)
print("Input a:")
print(a)
index2 = torch.tensor([0, 3, 2])
print("index:", index2)
out = torch.index_select(a, dim=0, index=index2)
print("Result out (selected rows in order index2):")
print(out)

### gather_examples
# torch.gather(input, dim, index) picks values from `input` along dimension `dim`.
# Formula (for dim=1 as example):
# out[i, j] = input[i, index[i, j]]
a = torch.tensor([[10, 11, 12],
                  [20, 21, 22]])  # (2, 3)
idx = torch.tensor([[2, 0],
                    [1, 1]])  # (2, 2)
print("\nExample 5 — torch.gather along dim=1 (pick columns per row):")
print("Input a:")
print(a)
print("index idx (column indices):")
print(idx)
out = torch.gather(a, dim=1, index=idx)
print("Result out (out[i, j] = a[i, idx[i, j]]):")
print(out)
# tensor([[12, 10],
#         [21, 21]])
