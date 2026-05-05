import torch

### clamp_demo
# Clamp formula (element-wise):
# y_i = min(max(x_i, min_value), max_value)
# Here min_value=1 and max_value=5, so each value is forced into [1, 5].
a = torch.rand(2, 2) * 10
print("Original tensor a (values roughly in [0, 10)):")
print(a)
a = a.clamp(1, 5)
print("\na.clamp(1, 5):")
print(a)
print("\nAfter clamp, every element satisfies: 1 <= a_ij <= 5")
