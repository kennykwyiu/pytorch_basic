# Math Symbols — Quick Guide

## What these symbols mean

- $\|x\|$ = “norm of x” (a kind of length/size)
- $\|\cdot\|$ = the norm *function* (`·` is a placeholder)
- $|a|$ = absolute value of a
- $\ge$ = greater than or equal to
- $\le$ = less than or equal to
- $\iff$ = “if and only if” (both directions are true)
- $x_1, x_2, ...$ = parts/components of x (subscripts)
- $\sum$ = sum (add many terms)
- $\sqrt{}$ = square root
- $p$ in $\|x\|_p$ = which norm rule you use (L1, L2, etc.)
- $\infty$ in $\|x\|_\infty$ = max absolute component

## Tiny examples

- If $x = (3, -4)$: $\|x\|_1 = |3| + |-4| = 7$, $\|x\|_2 = 5$, $\|x\|_\infty = 4$
- $\|x\| = 0 \iff x = 0$ means: only the zero vector has zero length

---

## Symbols used in “ML + PyTorch + Tensor — Summary”

### Variables & notation

- $x$ / $X$ = input (one sample vs a batch / dataset-style)
- $y$ = target/label
- $\hat{y}$ (written as **ŷ** in text) = predicted output
- $f(x; \theta)$ = model function with parameters $\theta$
- $\theta$ = model parameters (weights)
- $N$ = batch size (#samples)
- $D$ = input feature count
- $M$ = output feature count
- $()$ = shape parentheses, e.g. $(N, D)$

### Tensor / indexing symbols

- $x_i$ = the i-th element/component
- $\dots$ = “and so on” in sequences, e.g. $(x_1,\dots,x_n)$
- $#$ = “number of …” (count), e.g. “#dimensions”
- $\ne$ = not equal
- $\in$ = “in / belongs to”, e.g. $x \in [-1, 1]$

### Arithmetic operators

- $+$ add, $-$ subtract, $*$ multiply (elementwise), $/$ divide
- $**$ or $^$ power (exponent)
- $%$ remainder / modulo

### Comparison operators

- $==$ equal (elementwise in code)
- $!=$ not equal
- $>$ greater than, $<$ less than
- $\ge$ greater than or equal, $\le$ less than or equal

### Common “math function” symbols

- $\log$, $\log_2$, $\log_{10}$ = logs (natural, base-2, base-10)

- $\sqrt{\;}$ = square root
- $e$ = Euler’s number (used in $e^x$)
- $\sigma(x)$ = sigmoid function
- $\sin$, $\cos$, $\tan$ = trig functions
- $\tanh$ = hyperbolic tangent
- $\mathrm{abs}(x)$ or $|x|$ = absolute value

### Linear algebra / matrix symbols

- $@$ = matrix multiplication operator in Python
- $W^T$ = transpose (written as **Wᵀ** in text)
- Shapes rule: $(\dots, m, k) @ (\dots, k, n) \rightarrow (\dots, m, n)$

### Probability / distributions (from the distributions section)

- $\mathbb{E}[\;]$ = expectation (average under a distribution)
- $\nabla_\theta$ = gradient with respect to $\theta$
- $\log p_\theta(x)$ = log-probability
- $\mathcal{N}(\mu, \sigma^2)$ = Normal distribution
- $\mu$ = mean, $\sigma$ = standard deviation, $\epsilon$ = noise variable
- $\mathrm{KL}(P\|Q)$ = KL divergence