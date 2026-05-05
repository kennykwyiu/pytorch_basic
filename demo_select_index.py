import torch

# torch.where(condition, x, y) works element-wise:
# out[i] = x[i] if condition[i] is True, else y[i]
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

a = torch.rand(4,4)
b = torch.rand(4,4)

print("\nExample 2 — 2D where with condition a > 0.5:")
print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)

out = torch.where(a>0.5, a, b)
print("\nCondition (a > 0.5) — True means pick a, False means pick b:")
print(a > 0.5)
print("\nResult out = torch.where(a > 0.5, a, b):")
print(out)