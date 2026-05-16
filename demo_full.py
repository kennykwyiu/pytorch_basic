import torch

### full_create_constant_tensor
print("full_create_constant_tensor")

# Create a 2x3 tensor where every element is the constant value 10.
# torch.full(shape, fill_value)
a = torch.full((2, 3), 10)

print("a (2x3 filled with 10):")
print(a)

print("\na.shape (expected torch.Size([2, 3])):")
print(a.shape)
