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

### standard_normal_example
# Generate 5 samples from N(0, 1).
torch.manual_seed(0)
x = torch.normal(mean=0.0, std=1.0, size=(5,))
print("\n5 samples from N(0, 1):")
print(x)

### large_sample_distribution_check
# With many samples, sample mean/std should be close to target mean/std.
torch.manual_seed(0)
x = torch.normal(mean=10.0, std=2.0, size=(100000,))
print("\nLarge-sample check for N(10, 2):")
print("Sample mean (close to 10):")
print(x.mean().item())  # ~10
print("Sample std (close to 2):")
print(x.std().item())   # ~2
