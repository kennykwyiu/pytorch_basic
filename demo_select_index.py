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

### gather_with_dim0
# Formula (for dim=0):
# out[i, j] = input[index[i, j], j]
a = torch.linspace(1, 16, 16).view(4, 4)
print("\nExample 6 — torch.gather along dim=0 (pick rows per column):")
print("Input a:")
print(a)

idx2 = torch.tensor(
    [[0, 1, 1, 1],
     [0, 1, 2, 2],
     [0, 1, 3, 3]]
)
print("index idx2 (row indices):")
print(idx2)
out2 = torch.gather(a, dim=0, index=idx2)
print("Result out2 (out2[i, j] = a[idx2[i, j], j]):")
print(out2)

# dim=0, out[i, j, k] = input[index[i, j, k], j, k]
# dim=1, out[i, j, k] = input[i, index[i, j, k], k]
# dim=2, out[i, j, k] = input[i, j, index[i, j, k]]


### masked_select_example
# torch.masked_select(input, mask) returns a 1D tensor of elements where mask is True.
# Formula (conceptual): out = { input[i] | mask[i] == True }  (flattened in row-major order)
a = torch.linspace(1, 16, 16).view(4, 4)
mask = torch.greater(a, 8)
print("\nExample 7 — torch.masked_select (select elements > 8):")
print("Input a:")
print(a)
print("\nMask (a > 8):")
print(mask)
out = torch.masked_select(a, mask)
print("\nResult out = torch.masked_select(a, mask):")
print(out)

### take_example
# torch.take(input, index) treats `input` as a 1D flattened array and selects elements by index.
# Formula:
# out[i] = input.flatten()[index[i]]
a = torch.linspace(1, 16, 16).view(4, 4)
index3 = torch.tensor([0, 15, 13, 10])
print("\nExample 8 — torch.take (take elements by flattened indices):")
print("Input a:")
print(a)
print("flattened index:", index3)
b = torch.take(a, index=index3)
print("Result b = a.flatten()[index3]:")
print(b)

### nonzero_example
# torch.nonzero(input) returns the indices of all non-zero elements.
# For a 2D tensor:
# out[k] = [row_k, col_k] such that input[row_k, col_k] != 0
a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
print("\nExample 9 — torch.nonzero (indices of non-zero elements):")
print("Input a:")
print(a)
out = torch.nonzero(a)
print("Result out (indices where a != 0):")
print(out)
