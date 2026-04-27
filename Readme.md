## Machine Learning problem: 5 core elements

1. **Samples (Data / Dataset)**
    - What it is: examples the system learns from (inputs **X** and often targets/labels **y**).
    - Typical structure:
        - Features: numeric columns, tokens, pixels, etc.
        - Labels (supervised): class/value to predict; may be absent in unsupervised learning.
    - Common practice:
        - Train / validation / test split
        - Cleaning, normalization, feature engineering, augmentation
2. **Model**
    - What it is: a function that maps inputs to outputs, e.g. **ŷ = f(x; θ)**.
    - Contains:
        - **Parameters (θ)**: learned from data (weights).
        - **Hyperparameters**: chosen by you (learning rate, depth, regularization, etc.).
3. **Training**
    - What it is: learning parameters θ that minimize an objective on training data.
    - Key pieces:
        - Loss function (e.g., MSE, cross-entropy)
        - Optimizer (e.g., SGD, Adam)
        - Regularization (L2, dropout, early stopping) to reduce overfitting
4. **Inference (Prediction)**
    - What it is: using the trained model to produce outputs on new inputs.
    - Key concerns:
        - Correct, consistent preprocessing
        - Latency/throughput (real-time vs batch)
        - Post-processing (thresholds, decoding, rules)
5. **Testing (Evaluation)**
    - What it is: measuring performance, ideally on unseen test data.
    - Typical metrics:
        - Classification: accuracy, precision/recall, F1, ROC-AUC
        - Regression: MAE, RMSE, R²
    - Practice: error analysis, slice metrics, robustness checks, production monitoring

---

## PyTorch: 3 basic concepts (and how they connect)

### 1) Tensor

- **Definition:** PyTorch’s primary data structure — a multi-dimensional array (like NumPy, but optimized and can run on GPU).
- Key properties:
    - **rank / #dimensions** (0D, 1D, 2D, 3D+)
    - **shape** (size along each axis)
    - **dtype** (`float32`, `int64`, ...)
    - **device** (`cpu`, `cuda`)
- Typical ML shapes:
    - Tabular batch: `(N, D)`
    - Image (PyTorch often channels-first): `(C, H, W)`
    - Image batch: `(N, C, H, W)`

### 2) Autograd (Variable)

- Modern PyTorch: **Variable is merged into Tensor**.
- If `requires_grad=True`, PyTorch:
    - records operations (computational graph)
    - computes gradients with `loss.backward()`
- In inference, typically disable gradient tracking:
    - `with torch.no_grad(): ...`

### 3) nn.Module

- **Definition:** the base class for models/layers (e.g., `nn.Linear`, `nn.Conv2d`, custom networks).
- Provides:
    - `forward()` definition
    - parameter & submodule registration
    - mode switching: `.train()` / `.eval()`
    - moving to device: `.to(device)`
    - saving/loading via `state_dict()`

### How the circles overlap (intuition)

- Inputs/outputs of `nn.Module` are **tensors**
- Model parameters inside modules are **tensors**
- **autograd** computes gradients through tensor operations performed in the module

---

## Tensor basics: scalar → vector → matrix → (higher-order) tensor

- **Scalar** = 0D tensor, shape `()`
- **Vector** = 1D tensor, shape `(n,)`
- **Matrix** = 2D tensor, shape `(rows, cols)`
- **Tensor (3D+)** = 3 or more dimensions, e.g.:
    - `(C, H, W)` for a single image
    - `(N, C, H, W)` for a batch of images

> Mental model: “How many indices do I need to locate an element?”
> 

> 0 indices → scalar; 1 → vector; 2 → matrix; 3+ → tensor.
> 

---

## Linear model view: tensor math as a model

A common model computation is:

`Y = W X + b` (or in batch form `Y = X Wᵀ + b`)

- **X**: input tensor (samples/features)
- **W**: weight tensor (learnable parameters)
- **b**: bias tensor (learnable parameters)
- **Y**: output tensor (predictions)

### Typical shapes (batch form)

Let batch size `N`, input features `D`, output features `M`:

- `X`: `(N, D)`
- `W`: `(M, D)`
- `b`: `(M,)` (broadcasted across the batch)
- `Y`: `(N, M)`

### PyTorch mapping

`nn.Linear(D, M)` implements this:

- `Y = layer(X)` is effectively `Y = X @ W.T + b`
- `W` and `b` are tensors with gradients enabled during training

---

## Tensor basics in PyTorch: types, creation, attributes, ops, manipulation, NumPy conversion

### 1) Types (dtype)

- Common dtypes (matches the “Tensor types” slide):
    - **32-bit floating point**: `torch.float32` (alias `torch.float`) — common default for model weights
    - **64-bit floating point**: `torch.float64` (alias `torch.double`) — higher precision, slower/more memory
    - **16-bit floating point**: `torch.float16` (alias `torch.half`) — used for mixed precision on GPU
    - **8-bit integer (unsigned)**: `torch.uint8` — legacy / image bytes (often prefer `torch.int` + scaling for models)
    - **8-bit integer (signed)**: `torch.int8`
    - **16-bit integer (signed)**: `torch.int16` (alias `torch.short`)
    - **32-bit integer (signed)**: `torch.int32` (alias `torch.int`)
    - **64-bit integer (signed)**: `torch.int64` (alias `torch.long`) — very common for class labels / indices
    - **Boolean**: `torch.bool` — masks
- Notes / why it matters:
    - Most neural network computations use float tensors (`float32`, sometimes `float16/bfloat16`).
    - Many classification losses expect labels as `torch.long` (int64).

### 2) Creation

- Common creation APIs (matches the “Tensor creation” slide):
    - Base constructors:
        - `torch.Tensor(*size)` → uninitialized (values are whatever is in memory; not recommended unless you immediately fill it)
        - `torch.Tensor(data)` → create from data (similar to `np.array(...)`)
        - Prefer `torch.tensor(data)` when you want a clear “from data” constructor.
    - Constant tensors:
        - `torch.ones(*size)` → all 1s
        - `torch.zeros(*size)` → all 0s
        - `torch.eye(n)` (or `torch.eye(n, m)`) → identity / diagonal=1, else 0
    - Ranges:
        - `torch.arange(start, end, step)` → from `start` to `end` (end excluded) with step `step`
        - `torch.linspace(start, end, steps)` → evenly split `[start, end]` into `steps` values
    - Random:
        - `torch.rand(*size)` → uniform distribution on `[0, 1)`
        - `torch.randn(*size)` → standard normal distribution (mean 0, std 1)
        - `torch.normal(mean, std, size=...)` → normal with given mean/std
        - `torch.empty(*size).uniform_(from, to)` → uniform on `[from, to)` (in-place fill)
        - `torch.randperm(m)` → random permutation of `0..m-1`
- Practical tips:
    - Most code uses `torch.tensor(...)`, `zeros/ones`, `rand/randn`, `arange/linspace`.
    - Add `dtype=...` and `device=...` when needed: `torch.zeros(2, 3, dtype=torch.float32, device="cuda")`.

### 3) Attributes

- Commonly-used attributes:
    - `x.shape` / `x.size()` — shape
    - `x.ndim` — rank / #dimensions
    - `x.dtype` — **data type** (`torch.dtype`)
    - `x.device` — **device** (`torch.device`, e.g. `cpu`, `cuda:0`)
    - `x.layout` — **memory layout** (`torch.layout`, usually `torch.strided`)
    - `x.requires_grad` — whether autograd tracks this tensor
    - `x.grad` — gradient after `backward()` (if applicable)

### 4) Computation / ops

- Elementwise: `+ - * /`, `torch.exp`, `torch.log`, `torch.relu`, ...
- Matrix multiply: `@`, `torch.matmul`, `torch.mm`
- Reductions: `sum`, `mean`, `max`, `argmax`
- Notes:
    - Broadcasting follows NumPy-style rules
    - If `requires_grad=True`, these ops are recorded for backprop

### 5) Manipulation (shape/axis/indexing)

- Reshape/view: `reshape`, `view`
- Reorder axes: `transpose`, `permute`
- Add/remove dimensions: `unsqueeze`, `squeeze`
- Combine/split: `cat`, `stack`, `split`, `chunk`
- Indexing/slicing/masking: `x[0]`, `x[:, 10:20]`, `x[mask]`

### 6) Conversion with NumPy

- NumPy → Tensor: `torch.from_numpy(arr)`
- Tensor → NumPy: `tensor.numpy()` (tensor must be on CPU)
- Important:
    - `from_numpy` often shares memory with the NumPy array
    - For GPU tensors: `tensor.detach().cpu().numpy()`

---

## Tensor attributes (slide): `torch.dtype`, `torch.device`, `torch.layout`

- **`torch.dtype`**: what kind of numbers are stored (precision/bit-width), e.g. `torch.float32`, `torch.int64`, `torch.bool`.
- **`torch.device`**: where the tensor is stored, e.g. `torch.device("cpu")`, `torch.device("cuda:0")`.
    - Most ops require tensors to be on the **same device**.
- **`torch.layout`**: how the tensor is stored internally.
    - Most tensors are dense: `torch.strided` (default).
    - Sparse layouts exist (advanced) and support a different set of ops.

Example:

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32, device=torch.device("cpu"))
print(x.dtype, x.device, x.layout)
```

---

## Sparse tensors (slide): `torch.sparse_coo_tensor` (COO format)

### What is a sparse tensor?

- A **sparse tensor** is a tensor where **most elements are 0**.
- Instead of storing every element (including lots of zeros), sparse formats store only:
    - **indices** (coordinates of non-zero entries)
    - **values** (the non-zero values)
- Benefit: can save **memory** (and sometimes speed) when data is very sparse.

### COO (Coordinate) format in PyTorch

Create a COO sparse tensor with:

- `indices`: a 2D tensor of shape `(ndim, nnz)` where `nnz` is the number of non-zero elements
- `values`: a 1D tensor of shape `(nnz,)`
- `size`: the overall tensor shape

### Slide example (explained)

```python
indices = torch.tensor([[0, 1, 1],
                        [2, 0, 2]])
values  = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(indices, values, [2, 4])
```

- Shape: `x` is `(2, 4)`
- Non-zero entries:
    - `(0, 2) = 3`
    - `(1, 0) = 4`
    - `(1, 2) = 5`
- Dense view (for intuition):

```
[[0, 0, 3, 0],
 [4, 0, 5, 0]]
```

### Practical notes

- Sparse tensors are useful for very sparse data (e.g., one-hot/bag-of-words, large graphs).
- Operator support is more limited than dense tensors.
- Convert to dense for debugging: `x.to_dense()`

---

## Tensor arithmetic ops (slides): add / sub / mul / div, and matrix multiplication

### Overview

- **Elementwise ops (四则运算)**: add, subtract, multiply, divide (supports broadcasting)
- **Matrix multiplication (矩阵运算)**: `mm`, `matmul`, `@` (different from elementwise `*`)

### Addition (加法)

Equivalent forms:

```python
c = a + b
c = torch.add(a, b)
c = a.add(b)
```

In-place (modifies `a`):

```python
a.add_(b)
```

### Subtraction (减法)

```python
c = a - b
c = torch.sub(a, b)
c = a.sub(b)
```

In-place:

```python
a.sub_(b)
```

### Multiplication (乘法) — Hadamard / elementwise

This is **elementwise multiplication** (哈达玛积), not matrix multiply:

```python
c = a * b
c = torch.mul(a, b)
c = a.mul(b)
```

In-place:

```python
a.mul_(b)
```

### Division (除法)

```python
c = a / b
c = torch.div(a, b)
c = a.div(b)
```

In-place:

```python
a.div_(b)
```

### Matrix multiplication (矩阵乘法): `mm`, `matmul`, `@`

#### 2D matrices

- `torch.mm(a, b)` → **2D only**
- `torch.matmul(a, b)` / `a @ b` → works for vectors, matrices, and higher-dim tensors

Common equivalent forms (2D case):

```python
torch.mm(a, b)
torch.matmul(a, b)
a @ b
a.mm(b)
a.matmul(b)
```

#### High-dim tensors (dim > 2): use `matmul` / `@`

For `dim > 2`, matrix multiplication happens on the **last two dims** of each tensor, and all earlier dims are treated as **batch dims**.

- Think: do many independent matrix multiplies at once.
- Rule of thumb:
    - `(..., m, k) @ (..., k, n) -> (..., m, n)`
    - The `...` parts must match or be broadcastable.

Example (step-by-step):

```python
a = torch.ones(1, 2, 3, 4)   # a.shape = (B1, B2, m, k)
b = torch.ones(1, 2, 4, 3)   # b.shape = (B1, B2, k, n)
out = a @ b                  # matmul over the last 2 dims
```

1. **Split each shape into batch dims + matrix dims**
    - `a.shape = (1, 2, 3, 4)` → batch dims = `(1, 2)`, matrix dims = `(m, k) = (3, 4)`
    - `b.shape = (1, 2, 4, 3)` → batch dims = `(1, 2)`, matrix dims = `(k, n) = (4, 3)`
2. **Check the “inner” dims match**
    - `k` must match: `4` (from `a`) equals `4` (from `b`) ✅
3. **Compute the result matrix dims (the outcome)**
    - Matrix-multiply the last two dims: `(m, k) @ (k, n) -> (m, n)`
    - Here: `(3, 4) @ (4, 3)` gives `(3, 3)` because the inner `4` matches and the outer dims remain
4. **Broadcast / keep the batch dims**
    - Batch dims `(1, 2)` match exactly, so they stay the same.
5. **Combine batch dims + result matrix dims**
    - Output shape = `(1, 2, 3, 3)`

### Quick reminders

- `*` is elementwise; `@` is matrix multiply.
- Elementwise ops require same shape or broadcastable shapes.

---

## Other math ops (slides): logarithm and square root

### Logarithms (对数运算)

- `torch.log(a)` → natural log (base *e*)
- `torch.log2(a)` → log base 2
- `torch.log10(a)` → log base 10
- In-place version (modifies the original tensor): `torch.log_(a)`

Example:

```python
print(torch.log2(a))
print(torch.log10(a))
print(torch.log(a))
print(torch.log_(a))  # in-place
```

### Square root (开方运算)

- Out-of-place: `a.sqrt()` (returns a new tensor)
- In-place: `a.sqrt_()` (modifies `a`)

Example:

```python
print(a.sqrt())
print(a.sqrt_())  # in-place
print(a)          # a is changed
```

---

## Other math ops (slides): power and exponential

### Power (幂运算)

Equivalent forms:

```python
print(torch.pow(a, 2))
print(a.pow(2))
print(a**2)
```

In-place (modifies `a`):

```python
print(a.pow_(2))
```

### Exponential (指数运算)

- `torch.exp(a)` computes $e^a$ elementwise.
- In-place version: `a.exp_()` (modifies `a`)

Example:

```python
print(torch.exp(a))
b = a.exp_()  # in-place
```

---

## Broadcasting in PyTorch (slide)

### What broadcasting means

- **Broadcasting** is a rule that allows **elementwise operations** (such as `+`, `-`, `*`, `/`) on tensors with different shapes by **virtually expanding** one (or both) tensors to a common shape.
- This expansion is conceptual (often implemented via *views/strides*), so it typically **does not copy data**.

### The right-alignment rule (满足右对齐)

When checking if two shapes are broadcast-compatible:

1. Compare dimensions **from right to left** (the last dimension first).
2. For each dimension pair, they are compatible if:
    - sizes are **equal**, or
    - **one of them is 1** (can be expanded), or
    - one tensor “runs out” of leading dimensions (treat missing leading dims as **1**).

If any dimension pair fails these checks, broadcasting is not possible and PyTorch raises a shape mismatch error.

### Example from the slide

```python
a = torch.rand(2, 1, 1)
b = torch.rand(3)
out = a + b
```

Shapes:

- `a.shape = (2, 1, 1)`
- `b.shape = (3,)` → treat as `(1, 1, 3)` after right-alignment

Right-to-left check:

- last dim: `1` vs `3` → OK (1 expands to 3)
- next dim: `1` vs `1` → OK
- next dim: `2` vs `1` → OK (1 expands to 2)

Result:

- `out.shape = (2, 1, 3)`

### Practical notes

- Broadcasting applies to **elementwise** ops (`torch.add`, `torch.mul`, `a + b`, etc.), not matrix multiplication (`@` / `matmul`).
- If you see a size mismatch error, write the shapes and compare them **right-to-left**.

---

## Tensor rounding & remainder ops (slide): `floor`, `ceil`, `round`, `trunc`, `frac`, `%`

### Overview

These ops are useful when you need to:

- convert continuous values into “bucketed” / integer-like values
- extract integer vs fractional parts
- compute elementwise remainder (modulo)

Most of these functions return a float tensor (same dtype family) unless you explicitly cast to an integer dtype.

### `floor()` — round down (向下取整)

- Meaning: largest integer $\le x$ (elementwise).

```python
y = torch.floor(x)     # or x.floor()
```

Examples:

- `1.9 -> 1.0`
- `-1.2 -> -2.0`

### `ceil()` — round up (向上取整)

- Meaning: smallest integer $\ge x$ (elementwise).

```python
y = torch.ceil(x)      # or x.ceil()
```

Examples:

- `1.2 -> 2.0`
- `-1.2 -> -1.0`

### `round()` — round to nearest integer (四舍五入)

```python
y = torch.round(x)     # or x.round()
```

Note: tie-breaking for values exactly at `.5` can matter; check PyTorch docs/version behavior if your task is sensitive to `±0.5`.

### `trunc()` — truncate toward 0 (裁剪，只取整数部分)

- Meaning: drop the fractional part; rounds toward 0.

```python
y = torch.trunc(x)     # or x.trunc()
```

Examples:

- `1.9 -> 1.0`
- `-1.9 -> -1.0`

### `frac()` — fractional part only (只取小数部分)

- Meaning: keep only the fractional part.

```python
y = torch.frac(x)      # or x.frac()
```

Examples:

- `1.25 -> 0.25`
- `-1.25 -> -0.25`

### `%` / remainder (取余)

```python
r1 = x % y
r2 = torch.remainder(x, y)
```

Note: behavior for negative numbers depends on the exact remainder definition; `torch.remainder` matches Python-style remainder semantics.

---

## Tensor comparison ops (slide): `eq`, `equal`, `ge`, `gt`, `le`, `lt`, `ne`

### Overview

- These functions compare tensors and return **boolean** results.
- Most of them are **elementwise** comparisons and support **broadcasting**.
- They’re commonly used for building **masks** for indexing and filtering.

### `torch.eq(input, other)` / `==` (elementwise equality)

- Returns a **bool tensor** (elementwise).

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 0, 3])
print(a == b)         # tensor([ True, False,  True])
print(torch.eq(a, b)) # tensor([ True, False,  True])
```

### `torch.equal(tensor1, tensor2)` (full-tensor equality)

- Returns a **single Python bool**.
- `True` only if shape is the same **and** all elements are equal.

```python
print(torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2])))     # True
print(torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2, 3])))  # False
```

### `torch.ge / gt / le / lt / ne` (elementwise comparisons)

In `torch.ge`, `ge` is short for “greater than or equal to” (and similarly `gt` = greater than, `le` = less than or equal to, `lt` = less than, `ne` = not equal).

Elementwise comparisons (all return bool tensors):

- `torch.ge(a, b)` or `a >= b`
- `torch.gt(a, b)` or `a > b`
- `torch.le(a, b)` or `a <= b`
- `torch.lt(a, b)` or `a < b`
- `torch.ne(a, b)` or `a != b`

Example (masking):

```python
x = torch.tensor([-2.0, -0.5, 0.0, 1.2, 3.4])
mask = x > 0
print(mask)     # tensor([False, False, False,  True,  True])
print(x[mask])  # tensor([1.2000, 3.4000])
```
---

## Top-k / k-th value ops (slide): `sort`, `topk`, `kthvalue`

These ops help you find the **largest/smallest values** (and their **indices**) along a chosen dimension.

### `torch.sort(input, dim=-1, descending=False)`

- Sorts along `dim`.
- Returns `(values, indices)`:
    - `values`: sorted values
    - `indices`: where those values came from in the original tensor (along `dim`)

Example:

```python
x = torch.tensor([3.0, 1.0, 2.0])
values, idx = torch.sort(x)  # ascending
print(values)  # tensor([1., 2., 3.])
print(idx)     # tensor([1, 2, 0])
```

### `torch.topk(input, k, dim=-1, largest=True, sorted=True)`

- Gets the **largest k** values (or **smallest k** when `largest=False`) along `dim`.
- Returns `(values, indices)`.

Examples:

```python
x = torch.tensor([3.0, 1.0, 2.0])
values, idx = torch.topk(x, k=2)  # largest 2
print(values)  # tensor([3., 2.])
print(idx)     # tensor([0, 2])
```

Smallest k:

```python
values, idx = torch.topk(x, k=2, largest=False)
print(values)  # tensor([1., 2.])
print(idx)     # tensor([1, 2])
```

### `torch.kthvalue(input, k, dim=-1)`

- Returns the **k-th smallest** value along `dim` and its index.
- Important: `k` is **1-based** (`k=1` is the smallest).

Example:

```python
x = torch.tensor([3.0, 1.0, 2.0])
v, idx = torch.kthvalue(x, k=2)  # 2nd smallest
print(v)    # tensor(2.)
print(idx)  # tensor(2)
```

### Quick differences

- Need fully sorted output → `sort`
- Need only top/bottom k → `topk`
- Need only k-th smallest (a threshold/selection step) → `kthvalue`

---

## In-place operations in PyTorch (slide)

- **In-place / “就地” / “原位” operation**: modifies a tensor **in the same memory**, instead of creating a new tensor.
- Common patterns:
    - `x = x + y` (out-of-place idea) vs `x += y` / `x.add_(y)` (in-place)
    - Methods ending with `_` are usually in-place: `add_`, `sub_`, `mul_`, `div_`, `sqrt_`, `exp_`, `pow_`, `log_`, etc.

### Examples

Out-of-place (creates a new tensor):

```python
x = x + y
```

In-place (modifies `x`):

```python
x += y
x.add_(y)
```

### Why use in-place ops

- Potentially **saves memory** and can be **slightly faster** (fewer allocations).

### Important note (autograd)

- When training models, in-place ops on tensors involved in gradient computation can sometimes cause autograd errors (because old values may be needed for backward).