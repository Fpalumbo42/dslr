# Logistic Regression Training - Complete Guide

A detailed explanation of logistic regression, gradient descent, and one-vs-all classification from first principles.

---

## Table of Contents

1. [Overview](#overview)
2. [What is Logistic Regression?](#what-is-logistic-regression)
3. [Mathematical Components](#mathematical-components)
4. [One-vs-All Strategy](#one-vs-all-strategy)
5. [Complete Training Process](#complete-training-process)
6. [Practical Examples](#practical-examples)

---

## Overview

**Goal**: Classify students into 4 Hogwarts houses based on their academic scores.

**Method**: Logistic Regression with One-vs-All strategy

**Key Requirements**:
- ✅ Use gradient descent (batch, not stochastic)
- ✅ Implement everything from scratch (no sklearn for training)
- ✅ Achieve ≥98% accuracy

---

## What is Logistic Regression?

### The Core Idea

Logistic regression transforms a **linear combination** of features into a **probability** (0 to 1).

```
Linear combination → Sigmoid function → Probability → Classification
```

### Comparison with Linear Regression

**Linear Regression** (predicts continuous values):
```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
Output: any value from -∞ to +∞
```

**Logistic Regression** (predicts probabilities):
```
ŷ = sigmoid(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)
Output: probability between 0 and 1
```

---

## Mathematical Components

### 1. Sigmoid Function

**Formula:**
```
σ(z) = 1 / (1 + e^(-z))
```

**What it does:**
- Input: any real number z (from -∞ to +∞)
- Output: value between 0 and 1 (interpretable as probability)

**Properties:**
```
σ(0)    = 0.5      (neutral)
σ(+∞)   → 1        (very confident: class 1)
σ(-∞)   → 0        (very confident: class 0)
```

**Graph:**
```
  1.0 ┤          ╭─────────
      │        ╭╯
  0.5 ┤      ╭╯
      │    ╭╯
  0.0 ┤──╯
      └────────────────────
     -10  -5   0   5   10
```

**Examples:**
```python
σ(-5)  = 0.007    # 0.7% chance → class 0
σ(-1)  = 0.27     # 27% chance → class 0
σ(0)   = 0.5      # 50% chance → uncertain
σ(1)   = 0.73     # 73% chance → class 1
σ(5)   = 0.993    # 99.3% chance → class 1
```

**Why sigmoid?**
1. **Bounded**: Always outputs values in [0, 1]
2. **Smooth**: Differentiable everywhere (needed for gradient descent)
3. **Probabilistic**: Can interpret output as P(y=1|x)
4. **Non-linear**: Can model curved decision boundaries

---

### 2. Hypothesis Function

**Formula:**
```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

**Breaking it down:**

**Step 1: Linear combination**
```
z = θᵀx = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

where:
- θ = weight vector [θ₀, θ₁, θ₂, ..., θₙ]
- x = feature vector [1, x₁, x₂, ..., xₙ]  (x₀=1 is bias)
- z = weighted sum
```

**Step 2: Apply sigmoid**
```
h_θ(x) = σ(z)
```

**Interpretation:**
```
h_θ(x) ≥ 0.5  →  predict class 1
h_θ(x) < 0.5  →  predict class 0
```

**Concrete Example:**

Predict if a student is in Gryffindor:

```python
# Weights (learned during training)
θ = [0.5, -0.8, 1.2]  # [bias, θ_Astronomy, θ_Herbology]

# Student's normalized features
x = [1, 0.6, -0.3]  # [bias=1, Astronomy=0.6, Herbology=-0.3]

# Step 1: Linear combination
z = 0.5×1 + (-0.8)×0.6 + 1.2×(-0.3)
  = 0.5 - 0.48 - 0.36
  = -0.34

# Step 2: Sigmoid
h_θ(x) = 1 / (1 + e^(0.34))
       = 0.416

# Interpretation: 41.6% chance of Gryffindor
# Prediction: NOT Gryffindor (< 0.5)
```

---

### 3. Cost Function (Binary Cross-Entropy)

**Why do we need it?**

The cost function measures **how wrong** our predictions are. We want to find weights θ that **minimize** this error.

**Formula:**
```
J(θ) = -(1/m) × Σᵢ₌₁ᵐ [yⁱ log(h_θ(xⁱ)) + (1-yⁱ) log(1-h_θ(xⁱ))]

where:
- m = number of training examples
- yⁱ = actual label (0 or 1)
- h_θ(xⁱ) = predicted probability
```

**For a single example:**

```
If y = 1 (student IS in this house):
  cost = -log(h_θ(x))

  h_θ(x) = 1.0  →  cost = 0       (perfect!)
  h_θ(x) = 0.9  →  cost = 0.105   (good)
  h_θ(x) = 0.5  →  cost = 0.693   (uncertain)
  h_θ(x) = 0.1  →  cost = 2.303   (bad)
  h_θ(x) = 0.0  →  cost = ∞       (terrible!)

If y = 0 (student is NOT in this house):
  cost = -log(1 - h_θ(x))

  h_θ(x) = 0.0  →  cost = 0       (perfect!)
  h_θ(x) = 0.1  →  cost = 0.105   (good)
  h_θ(x) = 0.5  →  cost = 0.693   (uncertain)
  h_θ(x) = 0.9  →  cost = 2.303   (bad)
  h_θ(x) = 1.0  →  cost = ∞       (terrible!)
```

**Key insight:**
- Cost is **0** for perfect predictions
- Cost is **high** for confident wrong predictions
- Cost is **always ≥ 0**

**Example with 3 students:**

```python
y = [1, 0, 1]      # Actual labels
h = [0.9, 0.2, 0.6]  # Predictions

# Student 0: y=1, h=0.9
cost₀ = -log(0.9) = 0.105

# Student 1: y=0, h=0.2
cost₁ = -log(1-0.2) = -log(0.8) = 0.223

# Student 2: y=1, h=0.6
cost₂ = -log(0.6) = 0.511

# Total cost
J(θ) = (0.105 + 0.223 + 0.511) / 3 = 0.280
```

**Why this formula?**
1. **Convex**: Single global minimum (no local minima)
2. **Differentiable**: We can compute gradients
3. **Heavily penalizes confident mistakes**: Forces model calibration

---

### 4. Gradient

**What is it?**

The gradient tells us **which direction to adjust each weight** to reduce the cost.

**Formula:**
```
∂J(θ)/∂θⱼ = (1/m) × Σᵢ₌₁ᵐ (h_θ(xⁱ) - yⁱ) × xⱼⁱ
```

**Vectorized form (more efficient):**
```
∇J(θ) = (1/m) × Xᵀ(h - y)

where:
- X = feature matrix (m × n)
- h = predictions (m × 1)
- y = labels (m × 1)
- ∇J(θ) = gradient vector (n × 1)
```

**Intuition:**

Imagine the cost function as a 3D landscape. We want to find the lowest point (valley).

```
         ╱╲
        ╱  ╲
       ╱    ╲
  ────╱      ╲────
     ╱        ╲
    ╱     ★    ╲    ← We're here
   ╱  (minimum) ╲
  ╱              ╲
```

The gradient points **uphill** (steepest increase).
We go **downhill** (opposite direction) to minimize cost.

**Example:**

```python
# 3 students, 2 features (+ bias)
X = [[1,  0.5, -0.3],   # Student 0
     [1, -0.2,  0.8],   # Student 1
     [1,  0.9,  0.4]]   # Student 2

y = [1, 0, 1]  # Labels

θ = [0.1, 0.2, -0.5]  # Current weights

# Predictions
h = sigmoid(X @ θ) = [0.45, 0.40, 0.61]

# Errors
errors = h - y = [-0.55, 0.40, -0.39]

# Gradient for θ₀ (bias)
∂J/∂θ₀ = (1/3) × (1×(-0.55) + 1×0.40 + 1×(-0.39))
       = (1/3) × (-0.54)
       = -0.18

# Gradient for θ₁
∂J/∂θ₁ = (1/3) × (0.5×(-0.55) + (-0.2)×0.40 + 0.9×(-0.39))
       = -0.235

# Gradient for θ₂
∂J/∂θ₂ = (1/3) × ((-0.3)×(-0.55) + 0.8×0.40 + 0.4×(-0.39))
       = 0.110

∇J(θ) = [-0.18, -0.235, 0.110]
```

**Interpretation:**
- `∂J/∂θ₀ = -0.18` → Increase θ₀ to reduce cost (negative gradient)
- `∂J/∂θ₁ = -0.235` → Increase θ₁ to reduce cost
- `∂J/∂θ₂ = 0.110` → Decrease θ₂ to reduce cost (positive gradient)

---

### 5. Gradient Descent

**What is it?**

An iterative algorithm that adjusts weights to minimize the cost function.

**Algorithm:**
```
1. Initialize θ = [0, 0, ..., 0]
2. Repeat for N iterations:
   a. Compute predictions: h = sigmoid(X @ θ)
   b. Compute gradient: ∇J(θ) = (1/m) × Xᵀ(h - y)
   c. Update weights: θ := θ - α × ∇J(θ)
   d. (Optional) Compute cost J(θ)
3. Return θ
```

**Update Rule:**
```
θⱼ := θⱼ - α × ∂J(θ)/∂θⱼ

where α (alpha) = learning rate
```

**Learning Rate α:**

Controls the **step size**.

```
α too large:                α too small:              α just right:
  ╱╲              ╱╲           ╱╲                        ╱╲
 ╱  ╲    ★→←★    ╱  ╲         ╱  ╲ ★                    ╱  ╲  ★
╱    ╲──────────╱    ╲       ╱    ╲★                   ╱    ╲  ★
Overshoots!                  Very slow                        ★★
                                                            Converges
```

**Typical values:** 0.01, 0.1, 0.3, 1.0

**Example (one iteration):**

```python
# Current weights
θ = [0.1, 0.2, -0.5]

# Gradient (computed above)
∇J(θ) = [-0.18, -0.235, 0.110]

# Learning rate
α = 0.1

# Update
θ₀ = 0.1   - 0.1 × (-0.18)  = 0.118
θ₁ = 0.2   - 0.1 × (-0.235) = 0.2235
θ₂ = -0.5  - 0.1 × 0.110    = -0.511

# New weights
θ = [0.118, 0.2235, -0.511]

# Repeat 1000+ times until convergence
```

**Monitoring Progress:**

Track cost over iterations:

```
Iteration    0: J(θ) = 0.693  (random initialization)
Iteration  100: J(θ) = 0.420  (improving)
Iteration  500: J(θ) = 0.185  (improving)
Iteration 1000: J(θ) = 0.142  (converging)
Iteration 3000: J(θ) = 0.140  (converged!)
```

**Good sign:** Cost decreases monotonically ✓
**Bad sign:** Cost increases or oscillates → reduce α!

---

## One-vs-All Strategy

**Problem:** Logistic regression is binary (2 classes only).

**Our task:** Classify into 4 houses (multi-class).

**Solution:** Train **4 separate binary classifiers**.

### How It Works

**Training Phase:**

For each house h:
1. Create binary labels: `y_binary = 1 if house == h, else 0`
2. Train classifier θ_h using gradient descent
3. Store θ_h

**Example for Gryffindor:**

```python
Original:
Student 0: Gryffindor  →  y_binary = 1
Student 1: Hufflepuff  →  y_binary = 0
Student 2: Ravenclaw   →  y_binary = 0
Student 3: Gryffindor  →  y_binary = 1
Student 4: Slytherin   →  y_binary = 0

Train: θ_Gryffindor
```

**Prediction Phase:**

For a new student:
1. Compute probability for each house:
   ```python
   P(Gryffindor) = σ(θ_Gryffindor ᵀ x)
   P(Hufflepuff) = σ(θ_Hufflepuff ᵀ x)
   P(Ravenclaw)  = σ(θ_Ravenclaw ᵀ x)
   P(Slytherin)  = σ(θ_Slytherin ᵀ x)
   ```

2. Choose house with **highest probability**:
   ```python
   predicted_house = argmax(probabilities)
   ```

**Example:**

```python
x = [1, 0.8, -0.2, 0.5, 1.1]  # Student features

# Compute probabilities
P(Gryffindor) = σ(θ_G ᵀ x) = 0.77  ← Maximum!
P(Hufflepuff) = σ(θ_H ᵀ x) = 0.38
P(Ravenclaw)  = σ(θ_R ᵀ x) = 0.57
P(Slytherin)  = σ(θ_S ᵀ x) = 0.25

Prediction: Gryffindor (highest probability)
```

---

## Complete Training Process

### Full Pipeline

```
1. Load CSV
   ↓
2. Preprocessing:
   - Select features
   - Remove NaN rows
   - Normalize (z-score)
   - Add bias term
   ↓
3. One-vs-All Training:
   For each house h:
     - Create y_binary
     - Initialize θ_h = zeros
     - Run gradient descent
     - Store θ_h
   ↓
4. Save weights + normalization params
```

### Data Preprocessing

#### 1. Feature Selection

Choose features with good class separation:
```python
SELECTED_FEATURES = [
    'Astronomy',
    'Herbology',
    'Ancient Runes',
    'Divination',
    'Charms',
    'Flying'
]
```

**Why exclude some?**
- Defense Against the Dark Arts: correlated with Astronomy (r≈0.99)
- Care of Magical Creatures: homogeneous distribution
- Arithmancy: no discriminative power

#### 2. Handle Missing Values

```python
# Drop rows with any NaN
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# Result: 1600 → ~1500 students
```

#### 3. Normalization (Z-score)

**Formula:**
```
x_normalized = (x - μ) / σ
```

**Example:**
```python
# Raw Astronomy scores
X['Astronomy'] = [5.2, 10.8, -3.1, 7.5]

# Compute stats
μ = 5.0
σ = 3.0

# Normalize
X['Astronomy'][0] = (5.2 - 5.0) / 3.0 = 0.067
X['Astronomy'][1] = (10.8 - 5.0) / 3.0 = 1.93
X['Astronomy'][2] = (-3.1 - 5.0) / 3.0 = -2.7

# Result: mean ≈ 0, std ≈ 1
```

**Why normalize?**
1. **Faster convergence**: 10× fewer iterations
2. **Numerical stability**: Prevents overflow
3. **Fair features**: All contribute equally

**CRITICAL:** Save μ and σ for each feature!
```python
normalization_params = {
    'Astronomy': {'mean': 5.0, 'std': 3.0},
    'Herbology': {'mean': -2.1, 'std': 4.5},
    ...
}
```

You'll need these to normalize the test set!

#### 4. Add Bias Term

```python
# Before
X = [[0.5, -0.3, 1.2],
     [1.1,  0.5, -0.2]]

# After
X = [[1, 0.5, -0.3, 1.2],
     [1, 1.1,  0.5, -0.2]]
```

Why? Allows decision boundary to shift (not forced through origin).

---

## Practical Examples

### Example 1: Training Gryffindor Classifier

**Data:**
```python
X = [[1,  0.5, -0.3],   # 5 students
     [1, -0.8,  0.9],
     [1,  1.2,  0.4],
     [1, -0.2, -0.6],
     [1,  0.7,  1.1]]

y = ['Gryffindor', 'Hufflepuff', 'Gryffindor',
     'Ravenclaw', 'Gryffindor']

# Binary labels
y_binary = [1, 0, 1, 0, 1]
```

**Training:**
```python
θ = [0, 0, 0]  # Initialize
α = 0.1

# Iteration 1
h = sigmoid(X @ θ) = [0.5, 0.5, 0.5, 0.5, 0.5]
J(θ) = 0.693

errors = h - y_binary = [-0.5, 0.5, -0.5, 0.5, -0.5]
∇J(θ) = [-0.1, -0.04, 0.12]

θ = [0, 0, 0] - 0.1×[-0.1, -0.04, 0.12]
  = [0.01, 0.004, -0.012]

# ... repeat 1000 times

# Final result
θ_Gryffindor = [0.35, 0.82, -0.61]
J(θ) = 0.12  ← Much better!
```

### Example 2: Making a Prediction

```python
# New student
x = [1, 0.6, -0.2]

# Probabilities (using trained weights)
P(Gryffindor) = σ([0.35, 0.82, -0.61] @ x) = 0.72
P(Hufflepuff) = σ([-0.2, -0.5, 0.9] @ x)   = 0.34
P(Ravenclaw)  = σ([0.1, 0.3, 0.5] @ x)     = 0.45
P(Slytherin)  = σ([-0.3, -0.4, -0.8] @ x)  = 0.28

# Prediction
max_prob = 0.72
house = "Gryffindor"
```

---

## Implementation Pseudocode

### Main Training Function

```python
def train():
    # 1. Load and preprocess
    X, y, norm_params = preprocess_data(df)

    # 2. Train one-vs-all
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    all_thetas = {}

    for house in houses:
        # Binary labels
        y_binary = (y == house).astype(int)

        # Gradient descent
        theta = gradient_descent(X, y_binary, alpha=0.1, iterations=1000)

        # Store
        all_thetas[house] = theta

    # 3. Save weights
    save_weights(all_thetas, norm_params, 'weights.csv')
```

### Gradient Descent Function

```python
def gradient_descent(X, y, alpha=0.1, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for i in range(iterations):
        # Predictions
        h = sigmoid(X @ theta)

        # Gradient
        gradient = (1/m) * (X.T @ (h - y))

        # Update
        theta = theta - alpha * gradient

        # Monitor
        if i % 100 == 0:
            cost = compute_cost(X, y, theta)
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta
```

---

## Common Pitfalls & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **No normalization** | Doesn't converge, cost oscillates | Apply z-score normalization |
| **α too large** | Cost increases | Reduce learning rate (try 0.01) |
| **α too small** | Very slow training | Increase learning rate (try 0.3) |
| **Forgot bias** | Poor performance | Add column of 1s to X |
| **log(0) error** | NaN in cost | Use `np.clip(h, 1e-7, 1-1e-7)` |
| **Wrong dimensions** | Shape mismatch | Check: X(m,n), y(m,), θ(n,) |
| **Not saving norm params** | Wrong predictions | Save μ, σ for test set |

---

## Summary

### Key Formulas

```
Sigmoid:        σ(z) = 1 / (1 + e^(-z))
Hypothesis:     h_θ(x) = σ(θᵀx)
Cost:           J(θ) = -(1/m)Σ[y log(h) + (1-y)log(1-h)]
Gradient:       ∇J(θ) = (1/m)Xᵀ(h - y)
Update:         θ := θ - α∇J(θ)
Normalization:  x_norm = (x - μ) / σ
```

### Training Steps

```
1. Preprocess: Clean, normalize, add bias
2. For each house:
   - Create binary labels
   - Run gradient descent
   - Store weights
3. Save all weights + normalization params
```

### One-vs-All Prediction

```
1. For each house: compute P(house|x) = σ(θ_house ᵀ x)
2. Predict: argmax(probabilities)
```

---

**Now you understand the complete mathematics and process behind logistic regression training!** 🎓
