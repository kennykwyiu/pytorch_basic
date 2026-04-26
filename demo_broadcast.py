import torch

a = torch.rand(2,3)
b = torch.rand(3)
# a, 2*3
# b,   3
# -> 1*3
# c, 2*3
c = a + b

print("Tensor a (shape 2x3):")
print(a)
print("\nTensor b (shape 3, broadcast to 1x3 then 2x3):")
print(b)
print("\nBroadcast result c = a + b:")
print(c)
print("\nShape of c:")
print(c.shape)