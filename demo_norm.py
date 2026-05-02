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



### vector_norms
# L1 norm: ||x||_1 = sum_i |x_i|
# L2 norm: ||x||_2 = sqrt(sum_i x_i^2)
# Linf norm: ||x||_inf = max_i |x_i|
x = torch.tensor([3., -4.])
print("\nVector x:")
print(x)
print("\ntorch.norm(x, p=1)  # expected 7")
print(torch.norm(x, p=1))                 # 7  (L1)
print("\ntorch.norm(x, p=2)  # expected 5")
print(torch.norm(x, p=2))                 # 5  (L2)
print('\ntorch.norm(x, p=float("inf"))  # expected 4')
print(torch.norm(x, p=float("inf")))      # 4  (Linf)
