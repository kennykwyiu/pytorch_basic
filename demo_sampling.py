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
