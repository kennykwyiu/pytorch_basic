import torch

### concat_dim0
# torch.cat((a, b), dim=0) concatenates along rows (stack vertically).
# Shape rule (2D):
# if a is (m, n) and b is (k, n), then cat dim=0 -> (m + k, n)
a = torch.zeros((2, 4))
b = torch.ones((3, 4))

out = torch.cat((a, b), dim=0)
print("Example 1 — torch.cat dim=0 (vertical concat):")
print("Tensor a shape:", a.shape)
print(a)
print("\nTensor b shape:", b.shape)
print(b)
print("\nResult out = torch.cat((a, b), dim=0):")
print(out)
print("out.shape:")
print(out.shape)
