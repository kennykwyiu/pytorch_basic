import torch

# Tensor addition demo

a = torch.rand(2,3)
b = torch.rand(2,3)

print("Initial tensor a:")
print(a)
print("Initial tensor b:")
print(b)

print("\na + b:")
print(a + b)
print("\na.add(b) (out-of-place):")
print(a.add(b))
print("\na after out-of-place add (unchanged):")
print(a)
print("\ntorch.add(a, b):")
print(torch.add(a, b))
print("\na.add_(b) (in-place):")
print(a.add_(b))
print("\na after in-place add (updated):")
print(a)

# Tensor subtraction demo

print("\nCurrent tensor a before subtraction:")
print(a)
print("Current tensor b before subtraction:")
print(b)

print("\na - b:")
print(a - b)
print("\na.sub(b) (out-of-place):")
print(a.sub(b))
print("\na after out-of-place sub (unchanged):")
print(a)
print("\ntorch.sub(a, b):")
print(torch.sub(a, b))
print("\na.sub_(b) (in-place):")
print(a.sub_(b))
print("\na after in-place sub (updated):")
print(a)

# multiplication

print("\nCurrent tensor a before multiplication:")
print(a)
print("Current tensor b before multiplication:")
print(b)

print("\na * b:")
print(a * b)
print("\na.mul(b) (out-of-place):")
print(a.mul(b))
print("\na after out-of-place mul (unchanged):")
print(a)
print("\ntorch.mul(a, b):")
print(torch.mul(a, b))
print("\na.mul_(b) (in-place):")
print(a.mul_(b))
print("\na after in-place mul (updated):")
print(a)

# division

print("\nCurrent tensor a before division:")
print(a)
print("Current tensor b before division:")
print(b)

print("\na / b:")
print(a / b)
print("\na.div(b) (out-of-place):")
print(a.div(b))
print("\na after out-of-place div (unchanged):")
print(a)
print("\ntorch.div(a, b):")
print(torch.div(a, b))
print("\na.div_(b) (in-place):")
print(a.div_(b))
print("\na after in-place div (updated):")
print(a)