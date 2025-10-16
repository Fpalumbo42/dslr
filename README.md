# DSLR - Data Science × Logistic Regression

> *Harry Potter and the Data Scientist*

A machine learning project implementing logistic regression from scratch to sort Hogwarts students into their houses.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1: Data Analysis](#part-1-data-analysis)
- [Part 2: Data Visualization](#part-2-data-visualization)
- [Usage](#usage)

---

## Project Overview

The Sorting Hat has been bewitched! We must recreate its magic using machine learning to sort students into their houses based on their academic scores.

**Goal**: Implement a multi-class classifier using **one-vs-all logistic regression** with **gradient descent** to achieve ≥98% accuracy.

**Constraints**: 
- No pre-built functions for statistics (mean, std, percentile, etc.)
- No sklearn for training (only for evaluation)
- All mathematical operations must be implemented manually

---

## Part 1: Data Analysis

### `describe.py`

Displays statistical information for all numerical features (like pandas' `describe()`).

#### Implemented Statistics:

**Basic Statistics:**
- **Count**: Number of non-null values
  ```
  count = n (length of data)
  ```

- **Mean**: Average value
  ```
  mean = (Σ xi) / n
  ```

- **Standard Deviation**: Measure of spread
  ```
  std = √[Σ(xi - mean)² / (n-1)]
  ```
  *Note: Uses Bessel's correction (n-1) for sample std*

- **Min/Max**: Minimum and maximum values

**Percentiles** (25%, 50%, 75%):
- **50% (Median)**: Middle value when sorted
- Uses linear interpolation for non-integer positions:
  ```
  index = (percentile/100) × (n-1)
  if index is not integer:
      value = lower + weight × (upper - lower)
  ```

**Additional Statistics** (Bonus):
- **Range**: `max - min`
- **IQR** (Interquartile Range): `Q3 - Q1`
- **Variance**: `std²`
- **Skewness**: Measure of asymmetry
  ```
  skewness = (1/n) × Σ[(xi - mean) / std]³
  ```
- **Kurtosis**: Measure of "tailedness"
  ```
  kurtosis = [(1/n) × Σ[(xi - mean) / std]⁴] - 3
  ```
- **MAD** (Median Absolute Deviation): Robust measure of spread
  ```
  MAD = median(|xi - median(x)|)
  ```
- **Outliers**: Count of values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

---

## Part 2: Data Visualization

### `histogram.py`

**Question**: *Which Hogwarts course has a homogeneous score distribution between all four houses?*

- Creates histograms for all numerical features
- Each histogram shows all 4 houses overlaid with different colors
- Homogeneous distribution = similar spread across houses
- Helps identify features that don't strongly discriminate between houses

### `scatter_plot.py`

**Question**: *What are the two features that are similar?*

- Calculates **Pearson correlation coefficient** between all feature pairs
- Identifies the pair with highest correlation (most similar features)
- Displays only the most correlated pair

**Pearson Correlation Formula**:
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]

where:
- r ∈ [-1, 1]
- r = 1: perfect positive correlation
- r = -1: perfect negative correlation
- r = 0: no correlation
```

### `pair_plot.py`

**Question**: *Which features are you going to use for your logistic regression?*

- Creates a **scatter plot matrix** showing all feature combinations
- Diagonal: histograms of individual features
- Off-diagonal: scatter plots of feature pairs
- Helps visualize:
  - Feature separation by house (good for classification)
  - Redundant features (high correlation)
  - Feature distributions

**Selection criteria**:
- Features with good visual separation between houses
- Avoid highly correlated features (redundancy)
- Features with clear, non-overlapping distributions

---

## Usage

### Data Analysis
```bash
python src/describe.py
```

### Data Visualization
```bash
# Histogram - homogeneous distribution
python src/histogram.py

# Scatter plot - most similar features
python src/scatter_plot.py

# Pair plot - feature selection
python src/pair_plot.py
```

