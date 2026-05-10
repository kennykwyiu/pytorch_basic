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



### split_dim0_size3
print("\nsplit_dim0_size3")
# Split a 2D tensor into row chunks (dim=0).
# Here, `dim=0` means "split across rows", and `split_size=3` means each chunk has up to 3 rows.
a = torch.rand((3, 4))

print("Input tensor a (3x4):")
print(a)

# With 3 rows total, splitting by size 3 returns 1 chunk:
# - chunk 0 shape: [3, 4]
out = torch.split(a, 3, dim=0)

print("\nlen(out) (number of chunks returned by torch.split):")
print(len(out))

# Easy reading (no loop): print the whole result with a label.
print("\nout (tuple of chunks):")
print(out)

# Optional quick verification (still no loop): print chunk shapes.
print("\nchunk shapes (expected [(3, 4)]):")
print(tuple(t.shape for t in out))

### split_dim1_size3
print("\nsplit_dim1_size3")
# Split a 2D tensor into column chunks (dim=1).
# Here, `dim=1` means "split across columns", and `split_size=3` means each chunk has up to 3 columns.
a = torch.rand((10, 4))

print("Input tensor a (10x4):")
print(a)

# With 4 columns total, splitting by size 3 returns 2 chunks:
# - chunk 0 shape: [10, 3]
# - chunk 1 shape: [10, 1]
out = torch.split(a, 3, dim=1)

print("\nlen(out) (number of chunks returned by torch.split):")
print(len(out))

# If you want easy reading without looping, print the whole result with a label:
print("\nout (tuple of chunks):")
print(out)

# Optional (still no loop): print just the shapes so you can quickly verify the split:
print("\nchunk shapes (expected [(10, 3), (10, 1)]):")
print(tuple(t.shape for t in out))



### split_dim0_sizes_1_3_6
print("split_dim0_sizes_1_3_6")

# Split a 2D tensor into row chunks (dim=0) using explicit sizes.
# Here, `dim=0` means "split across rows", and the list [1, 3, 6] means:
# - first chunk gets 1 row
# - second chunk gets 3 rows
# - third chunk gets 6 rows
# The sizes must sum to the total number of rows (1 + 3 + 6 = 10).
a = torch.rand((10, 4))

print("Input tensor a (10x4):")
print(a)

out = torch.split(a, [1, 3, 6], dim=0)

print("\nlen(out) (number of chunks returned by torch.split):")
print(len(out))

# Debug: show expected shapes so it’s easy to verify the split at a glance.
print("\nchunk shapes (expected [(1, 4), (3, 4), (6, 4)]):")
print(tuple(t.shape for t in out))

# Print each chunk and its shape (per-chunk output is the point here).
for t in out:
	print(t, t.shape)