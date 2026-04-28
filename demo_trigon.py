import torch

### cosine_with_random_input
# Generate random values (in radians) and apply cosine element-wise.
a = torch.rand(2, 3)
b = torch.cos(a)

print("Input tensor a (random radians):")
print(a)
print("\ntorch.cos(a):")
print(b)

### cosine_with_zero_input
# cos(0) = 1, so every element in the output should be 1.
a = torch.zeros(2, 3)
b = torch.cos(a)

print("\nInput tensor a (all zeros):")
print(a)
print("\ntorch.cos(a) when a is zero:")
print(b)