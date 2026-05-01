## What these symbols mean

- `||x||` = “norm of x” (a kind of length/size)
- `||·||` = the norm *function* (`·` is a placeholder)
- `|a|` = absolute value of a
- `≥` = greater than or equal to
- `≤` = less than or equal to
- `⇔` = “if and only if” (both directions are true)
- `x₁, x₂, ...` = parts/components of x (subscripts)
- `Σ` / `sum` = sum (add many terms)
- `√` = square root
- `p` in `||x||ₚ` = which norm rule you use (L1, L2, etc.)
- `∞` in `||x||∞` = max absolute component

## Tiny examples

- If `x = (3, -4)`: `||x||₁ = |3| + |-4| = 7`, `||x||₂ = 5`, `||x||∞ = 4`
- `||x|| = 0 ⇔ x = 0` means: only the zero vector has zero length

---

## Symbols used in “ML + PyTorch + Tensor — Summary”

### Variables & notation

- `x` / `X` = input (one sample vs a batch / dataset-style)
- `y` = target/label
- `ŷ` = predicted output (“y-hat”)
- `f(x; θ)` = model function with parameters `θ`
- `θ` = model parameters (weights)

### Shapes / dimensions

- `N` = batch size (number of samples)
- `D` = input feature count
- `M` = output feature count
- `( )` = shape parentheses, e.g. `(N, D)`

### Tensor / indexing symbols

- `xᵢ` = the i-th element/component
- `…` = “and so on”
- `#` = “number of …” (count), e.g. “number of dimensions”
- `≠` = not equal
- `∈` = “in / belongs to”, e.g. `x ∈ [-1, 1]`

### Arithmetic operators

- `+` add, `-` subtract, `*` multiply (elementwise), `/` divide
- `**` or `^` power (exponent)
- `%` remainder / modulo

### Comparison operators

- `==` equal (elementwise in code)
- `!=` not equal
- `>` greater than, `<` less than
- `≥` greater than or equal, `≤` less than or equal

### Common math function symbols

- `log`, `log2`, `log10` = logs (natural, base-2, base-10)
- `√` = square root
- `e` = Euler’s number (used in `e^x`)
- `σ(x)` = sigmoid function
- `sin`, `cos`, `tan` = trig functions
- `tanh` = hyperbolic tangent
- `abs(x)` or `|x|` = absolute value

### Linear algebra / matrix symbols

- `@` = matrix multiplication operator in Python
- `Wᵀ` = transpose
- Shapes rule: `(..., m, k) @ (..., k, n) -> (..., m, n)`

### Probability / distributions (from the distributions section)

- `E[ ]` = expectation (average under a distribution)
- `∇θ` = gradient with respect to `θ`
- `log pθ(x)` = log-probability
- `N(μ, σ²)` = Normal distribution
- `μ` = mean, `σ` = standard deviation, `ε` = noise variable
- `KL(P||Q)` = KL divergence