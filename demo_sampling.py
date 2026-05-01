import torch

### normal_sampling_with_tensor_mean_std
# Sample from a normal distribution where mean/std are tensors (per-element parameters).
torch.manual_seed(1)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)
print("Mean tensor:")
print(mean)
print("\nStd tensor:")
print(std)
print("\nSample from torch.normal(mean, std):")
print(torch.normal(mean, std))

### manual_seed_reproducibility
# Same random seed => same random values; different seed => different values.
torch.manual_seed(123)
A = torch.rand(3)

torch.manual_seed(999)
B = torch.rand(3)

torch.manual_seed(123)
C = torch.rand(3)

print("\nTensor A (seed=123):")
print(A)
print("\nTensor B (seed=999):")
print(B)
print("\nTensor C (seed=123):")
print(C)
print("\nA and C are equal (expected True):")
print(torch.allclose(A, C))  # True
print("\nA and B are equal (expected False):")
print(torch.allclose(A, B))  # False
