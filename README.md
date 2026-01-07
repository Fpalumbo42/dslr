# DSLR - Logistic Regression from Scratch

Multi-class classifier (4 Hogwarts houses) using **one-vs-all logistic regression** with **gradient descent**.

**Goal**: ≥98% accuracy
**Dataset**: 1600 students (training), 400 students (test), 6 features, 4 classes

**Constraints**:
- ❌ Forbidden: mean, std, min, max, percentile, describe, etc.
- ❌ No sklearn for training (only accuracy_score for evaluation)
- ✅ Implement everything manually

---

## Part 1 & 2: Data Analysis & Visualization ✅

**Key formula - Standard Deviation** (used everywhere):
```
std = √[Σ(xi - mean)² / (n-1)]    # Bessel's correction (n-1)
```

**Visualization questions answered**:
1. **Histogram**: Which course is homogeneous? → **Care of Magical Creatures**
2. **Scatter Plot**: Which 2 features are similar? → **Astronomy & Defense Against the Dark Arts** (r ≈ 0.99)
3. **Pair Plot**: Which features to use? → See below

**Selected Features** (6):
```
✅ Astronomy         (excellent separation)
✅ Herbology         (excellent separation)
✅ Ancient Runes     (very good separation)
✅ Divination        (very good separation)
✅ Charms            (good separation)
✅ Flying            (good separation, complements others)
```

**Excluded Features**:
```
❌ Defense Against the Dark Arts  (multicollinearity with Astronomy, r ≈ 0.99)
❌ Care of Magical Creatures     (homogeneous distribution)
❌ Others                         (poor class separation)
```

---

## Part 3: Logistic Regression ⏳

### Overview

**Strategy**: One-vs-All (4 binary classifiers, one per house)

```
Training data: 1600 students × 6 features → 4 house classes
Test data:     400 students × 6 features → predict houses
```

### Step 1: Preprocessing

#### 1.1 Normalization (Z-Score)

**Why?** Features have different scales (Astronomy ~500, Herbology ~1). Without normalization, gradient descent converges slowly.

**Formula**:
```
x_norm = (x - μ) / σ
```

**Example** (using real values from logreg_model.json):

For Astronomy (μ = 39.47, σ = 521.50):
```
Student with score 600:  x_norm = (600 - 39.47) / 521.50 = 1.075
Student with score -200: x_norm = (-200 - 39.47) / 521.50 = -0.459
```

Result: All features have **mean=0, std=1** → fast convergence

#### 1.2 Bias Term

Add column of 1s at the beginning:
```
Original:  [x₁, x₂, x₃, x₄, x₅, x₆]
With bias: [1, x₁, x₂, x₃, x₄, x₅, x₆]  ← 7 values total
```

Purpose: Allows decision boundary to shift (not forced through origin)

---

### Step 2: One-vs-All Strategy

Train **4 separate binary classifiers**:

| Classifier      | Question             | Labels              |
|-----------------|----------------------|---------------------|
| θ_Gryffindor    | Is Gryffindor?       | Gryff=1, Others=0  |
| θ_Hufflepuff    | Is Hufflepuff?       | Huff=1, Others=0   |
| θ_Ravenclaw     | Is Ravenclaw?        | Rav=1, Others=0    |
| θ_Slytherin     | Is Slytherin?        | Sly=1, Others=0    |

Each classifier learns **θ** with 7 weights (1 bias + 6 features)

---

### Step 3: Core Math

#### 3.1 Sigmoid Function

**Formula**:
```
g(z) = 1 / (1 + e^(-z))
```

**Purpose**: Convert any number to probability [0, 1]

**Examples**:
```
z = 0   → g(0) = 0.5      (50%)
z = 2   → g(2) ≈ 0.88     (88%)
z = -2  → g(-2) ≈ 0.12    (12%)
z = 5   → g(5) ≈ 0.993    (99.3%)
z = -5  → g(-5) ≈ 0.007   (0.7%)
```

Behavior: Large positive z → 1, Large negative z → 0, z=0 → 0.5

#### 3.2 Hypothesis Function

**Formula**:
```
h(x) = sigmoid(θᵀ · x) = 1 / (1 + e^(-θᵀx))
```

**Example** (using real trained weights from logreg_model.json):

Gryffindor classifier θ = [-3.39, 1.27, -1.27, -2.23, 2.67]
Student features x = [1, 0.5, -0.3, 1.2, 0.8]

```
z = θᵀ · x = (-3.39×1) + (1.27×0.5) + (-1.27×-0.3) + (-2.23×1.2) + (2.67×0.8)
z = -3.39 + 0.635 + 0.381 - 2.676 + 2.136 = -2.914

h(x) = 1 / (1 + e^2.914) ≈ 0.05  →  5% probability Gryffindor
```

#### 3.3 Cost Function (Binary Cross-Entropy)

**Formula**:
```
J(θ) = -(1/m) × Σ[y·log(h(x)) + (1-y)·log(1-h(x))]
```

where m = 1600 students, y = actual label (0 or 1), h(x) = predicted probability

**Example** (3 students):

| Student   | y   | h(x) | Cost contribution    |
|-----------|-----|------|----------------------|
| Harry     | 1   | 0.95 | -log(0.95) = 0.051  |
| Hermione  | 1   | 0.70 | -log(0.70) = 0.357  |
| Draco     | 0   | 0.05 | -log(0.95) = 0.051  |

```
J(θ) = -(1/3) × [0.051 + 0.357 + 0.051] = 0.153
```

**Interpretation**:
- Lower cost = better model
- Perfect predictions → J(θ) = 0
- Wrong predictions → J(θ) increases

**Why log?** Penalizes confident wrong predictions heavily + creates convex optimization surface

---

### Step 4: Gradient Descent

#### 4.1 Gradient

**Formula**:
```
∇J(θ) = (1/m) × Xᵀ · (h - y)
```

where:
- X = feature matrix (1600 × 7)
- h = predictions vector
- y = labels vector

#### 4.2 Update Rule

**Formula**:
```
θ := θ - α × ∇J(θ)
```

where α = learning rate (0.1 in our implementation)

**Example iteration** (simplified, 1 weight):

```
Iteration 0:
  θ = 0 (initialization)
  J(θ) = 0.693
  ∇J = 0.25
  θ_new = 0 - 0.1 × 0.25 = -0.025

Iteration 1:
  θ = -0.025
  J(θ) = 0.680
  ∇J = 0.23
  θ_new = -0.025 - 0.1 × 0.23 = -0.048

... continues ...

Iteration 1000:
  θ = -3.39
  J(θ) = 0.020
  ∇J ≈ 0.000001  ← converged!
```

#### 4.3 Early Stopping

Stop when cost barely changes:
```python
if |J_previous - J_current| < tolerance (1e-6):
    break
```

**Example**:
```
Iteration 0    : Cost = 0.6931
Iteration 100  : Cost = 0.2451
Iteration 500  : Cost = 0.0453
Iteration 1200 : Cost = 0.0201
Iteration 1201 : Cost = 0.0201  ← difference < 1e-6
✓ Converged!
```

---

### Step 5: Making Predictions

For a new student in test set:

1. **Normalize** features using saved μ/σ from training
2. **Add bias** (1 at beginning)
3. **Compute probabilities** for each house:
   ```
   P(Gryffindor)  = sigmoid(θ_Gryff · x) = 0.85
   P(Hufflepuff)  = sigmoid(θ_Huff · x)  = 0.10
   P(Ravenclaw)   = sigmoid(θ_Rav · x)   = 0.03
   P(Slytherin)   = sigmoid(θ_Sly · x)   = 0.02
   ```
4. **Predict**: argmax([0.85, 0.10, 0.03, 0.02]) = **Gryffindor**

---

## Formula Summary

| Concept          | Formula                                        | Purpose                      |
|------------------|------------------------------------------------|------------------------------|
| Normalization    | `(x - μ) / σ`                                  | Scale features               |
| Sigmoid          | `1 / (1 + e^(-z))`                            | Convert to probability [0,1] |
| Hypothesis       | `h(x) = sigmoid(θᵀx)`                         | Predict probability          |
| Cost             | `J(θ) = -(1/m)Σ[y·log(h) + (1-y)·log(1-h)]` | Measure error                |
| Gradient         | `∇J = (1/m)Xᵀ(h - y)`                        | Direction to update θ        |
| Update           | `θ := θ - α·∇J`                               | Improve weights              |
| One-vs-All       | Train 4 binary classifiers                    | Handle 4 classes             |
| Prediction       | `argmax(probabilities)`                       | Choose most likely class     |

---

## Usage

```bash
# Part 1: Data Analysis
python src/data/describe.py datasets/dataset_train.csv

# Part 2: Visualization
python src/data/histogram.py
python src/data/scatter_plot.py
python src/data/pair_plot.py

# Part 3: Training & Prediction
python src/logistic_regression/logreg_train.py datasets/dataset_train.csv
python src/logistic_regression/logreg_predict.py datasets/dataset_test.csv logreg_model.json

# Evaluation
python evaluate.py
```
