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

---

## Trig & hyperbolic ops (slide): common usage cases

### Most commonly used (in everyday PyTorch work)

1. **`torch.abs`** — L1/MAE, error magnitudes, distances
2. **`torch.tanh`** — bounding outputs to `[-1, 1]` (common in some models / RL)
3. **`torch.sin` / `torch.cos`** — cyclic features and periodic patterns (time-of-day, angles, positional encodings)
4. **`torch.atan2`** — angle from a 2D vector `(x, y)` (geometry / vision gradients)

### `torch.sin`, `torch.cos`, `torch.tan` — periodic patterns

Usage cases:

- signals / waves (time series)
- cyclic features (hour, weekday) → represent as angles
- positional encodings / rotations (when your data involves angles)

Example (cyclic time feature):

```python
import torch, math

hour = torch.tensor([0., 6., 12., 18., 23.])
angle = 2 * math.pi * hour / 24
hour_sin = torch.sin(angle)
hour_cos = torch.cos(angle)
```

### `torch.asin`, `torch.acos`, `torch.atan` — inverse trig (recover an angle)

Usage cases:

- recover an angle after a geometric computation / normalization
- `atan(slope)` gives the angle of a line

Note: `asin/acos` inputs must be in `[-1, 1]`.

### `torch.atan2(y, x)` — angle from a 2D vector (quadrant-aware)

Usage cases:

- heading/direction from `(x, y)` (robotics/navigation)
- gradient orientation in CV (e.g., `atan2(dy, dx)`)

Example:

```python
import torch
y = torch.tensor([1.0, 1.0, -1.0])
x = torch.tensor([1.0, -1.0, 1.0])
angle = torch.atan2(y, x)  # radians
```

### `torch.sinh`, `torch.cosh`, `torch.tanh` — smooth nonlinearities

Usage cases:

- `tanh` as an activation / squashing function to bound outputs to `[-1, 1]`

Example (bounded outputs):

```python
import torch
raw = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
bounded = torch.tanh(raw)
```

### `torch.abs` — magnitudes and robust losses

Usage cases:

- L1/MAE: `abs(pred - target)`
- Manhattan distance

Example:

```python
import torch
pred = torch.tensor([2.0, 3.5])
target = torch.tensor([1.0, 4.0])
mae = torch.mean(torch.abs(pred - target))
```

### Common gotchas

- Trig functions use **radians** (not degrees).
- `asin/acos` require inputs in `[-1, 1]` (clamp if needed).

---

## Key math ops (slide): `abs`, `sigmoid`, `sign` (explained)

### `torch.abs(x)` — absolute value

Elementwise absolute value:

- `x > 0` → `abs(x) = x`
- `x < 0` → `abs(x) = -x`
- `x = 0` → `0`

Example:

```python
import torch
x = torch.tensor([-2.0, -0.5, 0.0, 3.0])
print(torch.abs(x))  # tensor([2.0000, 0.5000, 0.0000, 3.0000])
```

Common usage:

- L1/MAE loss: `torch.mean(torch.abs(pred - target))`
- magnitude/error size ignoring sign

### `torch.sigmoid(x)` — squashes values into (0, 1)

Definition:

$\sigma(x) = 1 / (1 + e^{-x})$

Intuition:

- very negative `x` → close to `0`
- `x = 0` → `0.5`
- very positive `x` → close to `1`

Example:

```python
import torch
x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
print(torch.sigmoid(x))  # ~tensor([0.0180, 0.2689, 0.5000, 0.7311, 0.9820])
```

Common usage:

- binary classification: logits → probability
- gates in neural nets (values between 0 and 1)

Training note: `BCEWithLogitsLoss` usually expects logits directly (sigmoid is applied internally for numerical stability).

### `torch.sign(x)` — sign function (−1, 0, +1)

Elementwise sign:

- `x > 0` → `+1`
- `x < 0` → `-1`
- `x = 0` → `0`

Example:

```python
import torch
x = torch.tensor([-2.0, -0.5, 0.0, 3.0])
print(torch.sign(x))  # tensor([-1., -1.,  0.,  1.])
```

Common usage (more niche):

- direction-only signals (positive vs negative)
- some optimization/regularization tricks

Autograd note: `sign()` is not smooth at 0, so it’s usually not used as a standard activation for backprop-trained models.

### Quick “most common” summary

- Most common overall: `abs`, `sigmoid`
- Useful but more specialized: `sign`

### Notes from the slide (“balance” annotations)

#### L1 vs L2 loss (intuition)

- **L1 loss**: $|x|$ (a “V” shape). Gradient magnitude is roughly constant away from 0 → can be more **robust to outliers**.
- **L2 loss**: $x^2$ (a “U” shape). Penalizes large errors more strongly → pushes big errors down harder, but is **more sensitive to outliers**.

#### Sigmoid curve (intuition)

- S-shaped curve mapping real numbers to `(0, 1)`.
- Key points:
    - `sigmoid(0) = 0.5`
    - large negative → close to `0`
    - large positive → close to `1`

#### `sign(x)` piecewise definition

- `sign(x) = -1` if `x < 0`
- `sign(x) = 0` if `x = 0`
- `sign(x) = 1` if `x > 0`

#### Other functions listed on the slide (quick meanings)

- `torch.erf(x)`: Gaussian error function (stats/probability; special function)
- `torch.erfinv(x)`: inverse error function
- `torch.neg(x)`: elementwise negation (`-x`)
- `torch.reciprocal(x)`: elementwise reciprocal (`1/x`)
- `torch.rsqrt(x)`: reciprocal square root (`1/sqrt(x)`)
- `torch.lerp(a, b, w)`: linear interpolation `a + w * (b - a)`
- `torch.addcdiv(input, tensor1, tensor2, value=...)`: `input + value * tensor1 / tensor2` (often seen in optimizer math)
- `torch.addcmul(input, tensor1, tensor2, value=...)`: `input + value * tensor1 * tensor2` (often seen in optimizer math)
- `torch.cumsum(x, dim=...)`: cumulative sum along a dimension
- `torch.cumprod(x, dim=...)`: cumulative product along a dimension

---

## Statistics & reductions (slides): quick summary + math examples

These two slides are about **statistics / reduction operations on tensors** in PyTorch (Tensor 中统计学相关的函数).

Use these when you want to “reduce” a tensor into summary values (mean/sum/max/...) or analyze distributions (std/var/histograms).

### Setup tensors used in examples

```python
import torch
x = torch.tensor([1., 2., 3., 4.])
y = torch.tensor([1., 2., 3.])
labels = torch.tensor([0, 2, 2, 1, 0, 2])
```

### `torch.mean(x)` — average

Math: `(1+2+3+4)/4 = 2.5`

```python
torch.mean(x)  # tensor(2.5000)
```

Usage cases:

- compute average loss over a batch (common: `loss.mean()`)
- feature normalization stats (mean of a feature dimension)

### `torch.sum(x)` — sum

Math: `1+2+3+4 = 10`

```python
torch.sum(x)  # tensor(10.)
```

Usage cases:

- total loss before averaging (e.g., sum over tokens)
- count items after a mask: `(mask).sum()`

### `torch.prod(x)` — product

Math: `1*2*3*4 = 24`

```python
torch.prod(x)  # tensor(24.)
```

Usage cases:

- multiply probabilities / factors
- compute a “total scale” from per-dim scales

### `torch.max(x)` / `torch.min(x)` — max/min value

Math: `max=4`, `min=1`

```python
torch.max(x)  # tensor(4.)
torch.min(x)  # tensor(1.)
```

Usage cases:

- find peak/lowest values (debugging activations, clipping, ranges)
- with `dim=...`: get per-row/per-feature max/min and indices

### `torch.argmax(x)` / `torch.argmin(x)` — index of max/min

0-based indices: `max` is at index `3`, `min` is at index `0`

```python
torch.argmax(x)  # tensor(3)
torch.argmin(x)  # tensor(0)
```

Usage cases:

- classification prediction from logits: `logits.argmax(dim=1)`
- pick best candidate/action with highest score

### `torch.var(y)` / `torch.std(y)` — variance / standard deviation

Math:

- mean: `(1+2+3)/3 = 2`
- squared deviations: `(1-2)^2=1`, `(2-2)^2=0`, `(3-2)^2=1`
- population variance: `(1+0+1)/3 = 2/3 ≈ 0.6667`
- population std: `sqrt(2/3) ≈ 0.8165`

Note: exact output can differ based on PyTorch’s correction/unbiased setting.

```python
torch.var(y)
torch.std(y)
```

Usage cases:

- standardization: `(x - mean) / std`
- monitor distribution shift / detect exploding activations

### `torch.median(x)` — median (middle value)

For `[1,2,3,4]`, median is `(2+3)/2 = 2.5`

```python
torch.median(x)  # tensor(2.5000)
```

Usage cases:

- robust “center” statistic when outliers exist
- robust thresholding (median-based filters)

### `torch.mode(z)` — mode (most frequent value)

```python
z = torch.tensor([1, 1, 2, 3, 3, 3])
torch.mode(z)  # mode value is 3
```

Usage cases:

- most common class/label in a window or batch
- majority vote aggregation

### `torch.bincount(labels)` — counts of each integer value

Counts: `0→2`, `1→1`, `2→3`

```python
torch.bincount(labels)  # tensor([2, 1, 3])
```

Usage cases:

- check class imbalance / label distribution
- count occurrences of discrete IDs

### `torch.histc(a, bins=..., min=..., max=...)` — histogram counts

```python
a = torch.tensor([0.1, 0.2, 0.9, 1.1, 1.9])
torch.histc(a, bins=2, min=0.0, max=2.0)  # ~tensor([3., 2.])
```

Usage cases:

- quick distribution check (e.g., activations, weights)
- detect saturation/clipping/outliers


---

## `torch.distributions` (slide): sampling + gradients (examples)

`torch.distributions` provides **parameterized probability distributions** plus common operations:

- `.sample()` draw samples (usually not differentiable through the sampled value)
- `.rsample()` draw samples using **reparameterization** (pathwise gradient; differentiable when supported)
- `.log_prob(x)` compute log probability $\log p(x)$
- `.entropy()` entropy (often used as a regularizer in RL)

### A) Score function estimator (得分函数) — policy gradient (RL)

Goal: optimize an expectation over samples:

$\mathbb{E}_{x \sim p_\theta}[f(x)]$

Using the score function trick:

$\nabla_\theta \mathbb{E}[f(x)] = \mathbb{E}[f(x)\nabla_\theta \log p_\theta(x)]$

Practical idea:

- sample an action
- compute `log_prob(action)`
- weight it by reward (or advantage)

Example (REINFORCE-style with `Categorical`):

```python
import torch
from torch.distributions import Categorical

logits = torch.tensor([0.2, 1.0, -0.5], requires_grad=True)  # policy parameters
dist = Categorical(logits=logits)

action = dist.sample()               # non-differentiable sample
logp = dist.log_prob(action)         # differentiable w.r.t. logits

reward = torch.tensor(2.0)           # example reward
loss = -reward * logp                # maximize reward => minimize -reward*logp

loss.backward()
print(logits.grad)
```

Notes:

- Often use a baseline to reduce variance: `-(reward - baseline) * logp`.

### B) Pathwise derivative estimator — reparameterization (VAE)

For some distributions (e.g., Normal), sampling can be written as:

- $\epsilon \sim \mathcal{N}(0, 1)$
- $z = \mu + \sigma \epsilon$

This lets gradients flow through $\mu, \sigma$.

Example (`Normal` + `.rsample()`):

```python
import torch
from torch.distributions import Normal

mu = torch.tensor(0.0, requires_grad=True)
sigma = torch.tensor(1.0, requires_grad=True)
dist = Normal(mu, sigma)

z = dist.rsample()                   # differentiable sample
loss = (z - 3.0) ** 2                # any differentiable objective

loss.backward()
print(mu.grad, sigma.grad)
```

Comparison: `.sample()` does *not* provide a pathwise gradient:

```python
mu = torch.tensor(0.0, requires_grad=True)
sigma = torch.tensor(1.0, requires_grad=True)
dist = Normal(mu, sigma)

z = dist.sample()                    # breaks pathwise gradient through z
loss = (z - 3.0) ** 2
loss.backward()
print(mu.grad, sigma.grad)
```

Rule of thumb:

- RL / non-differentiable environment → score function (`log_prob`)
- VAE / differentiable latent sampling → pathwise (`rsample`)

---

## `torch.distributions` (slide): KL divergence, transforms, constraints (use cases)

### 1) KL Divergence

What it is: a measure of how different two distributions are.

$\mathrm{KL}(P\|Q) = \mathbb{E}_{x\sim P}\left[\log \frac{P(x)}{Q(x)}\right]$

Use cases:

- **VAE**: KL regularizer between $q(z|x)$ and prior $p(z)$ (encourages latent space to match the prior).
- **RL (PPO/TRPO)**: control how much the policy changes (KL between old and new policy).

Example (`Normal` vs `Normal`):

```python
import torch
from torch.distributions import Normal, kl_divergence

p = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
q = Normal(loc=torch.tensor(1.0), scale=torch.tensor(2.0))

kl = kl_divergence(p, q)  # KL(p || q)
print(kl)
```

### 2) Transforms

What it is: build a new distribution by transforming samples from a base distribution.

Use cases:

- **Bounded continuous actions in RL**: sample a `Normal`, then apply `tanh` so actions are in `(-1, 1)`.
- **Change support**: map real values to positive (`ExpTransform`) or to `(0, 1)` (`SigmoidTransform`).
- **Normalizing flows** (advanced): compose transforms to get flexible distributions.

Example (Normal + `tanh`):

```python
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

base = Normal(loc=0.0, scale=1.0)
dist = TransformedDistribution(base, [TanhTransform()])

a = dist.rsample((5,))    # samples in (-1, 1)
logp = dist.log_prob(a)   # log prob under the transformed distribution
print(a, logp)
```

### 3) Constraints

What it is: distributions require valid parameters (e.g., `Normal(scale)` must be **positive**). Constraints help validate parameters and prevent silent bugs.

Use cases:

- **Debugging**: catch invalid model outputs early (negative std, invalid probabilities, etc.).
- **Model parameterization**: output unconstrained values, then convert them to valid parameters (`softplus`, `sigmoid`, etc.).

Example (make `scale` valid):

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal

raw_scale = torch.tensor(-0.7)               # unconstrained
scale = F.softplus(raw_scale) + 1e-6         # now > 0
dist = Normal(loc=0.0, scale=scale)
```

---

## `torch.distributions` (slide): distribution “cheat sheet” (with examples)

This is a more practical way to remember the long list of built-in distributions: match the **data type / support** to the right distribution, then use the common API (`sample/rsample`, `log_prob`, `entropy`).

### A) Binary outcomes (0/1)

**`Bernoulli`** — binary label, masking, click/no-click

```python
from torch.distributions import Bernoulli
dist = Bernoulli(probs=0.7)
x = dist.sample((5,))      # tensor of 0/1
lp = dist.log_prob(x)
```

### B) Multi-class (single class index)

**`Categorical`** — classification sampling, RL discrete actions

```python
import torch
from torch.distributions import Categorical
dist = Categorical(logits=torch.tensor([1.0, 0.5, -0.2]))
a = dist.sample()          # class index
lp = dist.log_prob(a)
```

### C) Multi-class (one-hot vector)

**`OneHotCategorical`** — need one-hot actions/labels directly

```python
import torch
from torch.distributions import OneHotCategorical
dist = OneHotCategorical(logits=torch.tensor([1.0, 0.5, -0.2]))
oh = dist.sample()         # one-hot vector
```

### D) Continuous real-valued (unbounded)

**`Normal`** (most common), **`Laplace`**, **`StudentT`**, **`Cauchy`** — noise models, robust modeling (heavy tails), VAE latents

```python
from torch.distributions import Normal
dist = Normal(loc=0.0, scale=1.0)
z = dist.rsample((10,))    # differentiable sample (pathwise)
lp = dist.log_prob(z)
```

### E) Positive continuous (> 0)

**`Exponential`**, **`Gamma`**, **`Weibull`**, **`LogNormal`**, **`HalfNormal`**, **`HalfCauchy`** — times, rates, scales, strictly positive variables

```python
from torch.distributions import Exponential
dist = Exponential(rate=2.0)
t = dist.sample((5,))      # positive samples
```

### F) Bounded continuous in [0, 1]

**`Beta`** — probabilities / proportions (CTR, uncertainty over a Bernoulli probability)

```python
from torch.distributions import Beta
dist = Beta(concentration1=2.0, concentration0=5.0)
p = dist.sample((5,))
```

### G) Count data (0,1,2,3,...)

**`Poisson`** — events per interval (arrivals, clicks per minute)

```python
from torch.distributions import Poisson
dist = Poisson(rate=3.0)
c = dist.sample((5,))
```

### H) Successes out of N trials

**`Binomial`** — number of successes in fixed trials

```python
from torch.distributions import Binomial
dist = Binomial(total_count=10, probs=0.3)
k = dist.sample((5,))
```

### I) Simplex (components sum to 1)

**`Dirichlet`** — topic proportions, mixture weights, distributions over categorical probabilities

```python
import torch
from torch.distributions import Dirichlet
dist = Dirichlet(torch.tensor([0.5, 1.0, 2.0]))
pi = dist.sample()         # sums to 1
```

### J) Multivariate continuous (correlated)

**`MultivariateNormal`** / **`LowRankMultivariateNormal`** — correlated Gaussian variables, covariance modeling

```python
import torch
from torch.distributions import MultivariateNormal
dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
x = dist.rsample((4,))
```

### K) Useful wrappers (appear in the list)

**`Independent`** — treat some batch dims as “event dims” so `log_prob` reduces correctly (common in VAEs).

```python
import torch
from torch.distributions import Normal, Independent
dist = Independent(Normal(torch.zeros(3), torch.ones(3)), 1)
x = dist.rsample()
lp = dist.log_prob(x)      # scalar per sample
```

**`TransformedDistribution`** — base distribution + transform (common in RL for bounded actions).

**`RelaxedBernoulli` / `RelaxedOneHotCategorical`** — differentiable (continuous) relaxations of discrete distributions (advanced).