import torch

# --- Create input tensor ---
# Create a 3x4 tensor of random values sampled uniformly from [0, 1).
a = torch.rand((3, 4))

# Debug: print a labeled summary so the output is easy to scan.
print("a shape:", a.shape, "dtype:", a.dtype, "device:", a.device)

# Debug: print the full tensor values (fine here since it’s small).
print("a values:\n", a)

# --- Chunk along rows (dim=0) ---
# Split `a` into 2 chunks along dim=0 (rows).
# With 3 rows, this yields 2 tensors: shapes [2,4] and [1,4].
out = torch.chunk(a, 2, dim=0)

# Debug: print a one-line shape summary (still loop-free in your code).
print("chunk(dim=0, chunks=2) out shapes:", tuple(t.shape for t in out))

# Debug: print the full tuple result with a label (loop-free).
print("chunk(dim=0, chunks=2) out:", out)

# --- Split into size-2 chunks along rows (dim=0) ---
# Split `a` into chunks of size 2 along dim=0 (rows).
# With 3 rows, this returns 2 tensors: shapes [2,4] and [1,4].
out = torch.split(a, 2, dim=0)

print("split(dim=0, size=2) out shapes:", tuple(t.shape for t in out))
print("split(dim=0, size=2) out:", out)

# --- Split into size-2 chunks along columns (dim=1) ---
# Split `a` into chunks of size 2 along dim=1 (columns).
# With 4 columns, this returns 2 tensors: shapes [3,2] and [3,2].
out = torch.split(a, 2, dim=1)

print("split(dim=1, size=2) out shapes:", tuple(t.shape for t in out))
print("split(dim=1, size=2) out:", out)

# --- Split into size-3 chunk along rows (dim=0) ---
# Split `a` into chunks of size 3 along dim=0 (rows).
# Since `a` has exactly 3 rows, this returns 1 tensor: shape [3,4].
out = torch.split(a, 3, dim=0)

print("split(dim=0, size=3) out shapes:", tuple(t.shape for t in out))
print("split(dim=0, size=3) out:", out)