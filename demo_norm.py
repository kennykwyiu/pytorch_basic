import torch

### lp_distance_between_tensors
# Formula: dist_p(a, b) = (sum_i |a_i - b_i|^p)^(1/p)
a = torch.rand(2, 1)
b = torch.rand(2, 1)

print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)
print("\ntorch.dist(a, b, p=1)  # L1 distance: sum_i |a_i - b_i|")
print(torch.dist(a, b, p=1))
print("\ntorch.dist(a, b, p=2)  # L2 distance: sqrt(sum_i (a_i - b_i)^2)")
print(torch.dist(a, b, p=2))
print("\ntorch.dist(a, b, p=3)  # L3 distance: (sum_i |a_i - b_i|^3)^(1/3)")
print(torch.dist(a, b, p=3))

print(torch.norm(a))
print(torch.norm(a, p=1))
