# DSLR - Data Science × Logistic Regression

> *Harry Potter and the Data Scientist*

A machine learning project implementing logistic regression from scratch to sort Hogwarts students into their houses.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1: Data Analysis](#part-1-data-analysis)
- [Part 2: Data Visualization](#part-2-data-visualization)
  - [Histogram Analysis](#histogram-analysis)
  - [Scatter Plot Analysis](#scatter-plot-analysis)
  - [Pair Plot Analysis](#pair-plot-analysis)
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

### Histogram Analysis

**Question**: *Which Hogwarts course has a homogeneous score distribution between all four houses?*

#### Purpose
Histograms help identify which features have similar distributions across all houses, indicating they may not be useful for distinguishing between houses.

#### Method
- Creates overlaid histograms for all numerical features
- Each histogram shows all 4 houses with different colors:
  - 🔴 Gryffindor (red)
  - 🟡 Hufflepuff (yellow)  
  - 🔵 Ravenclaw (blue)
  - 🟢 Slytherin (green)

#### Results & Analysis

![Histogram Comparison](readme_images/histograms.png)

**Answer: Care of Magical Creatures**

**Reasoning:**
Looking at the histogram comparison, **Care of Magical Creatures** shows the most homogeneous distribution because:
- All four house histograms overlap significantly
- The distribution shapes are nearly identical across houses
- The mean scores are centered around the same range (approximately -1 to 1)
- No house shows a distinctly different pattern

**Other observations:**
- **Arithmancy**: Also shows high overlap, making it a poor discriminator
- **Astronomy**: Shows excellent separation with distinct peaks for each house
- **Defense Against the Dark Arts**: Clear separation between houses
- **Charms**: Distinct distributions, especially Hufflepuff (yellow) is well separated

---

### Scatter Plot Analysis

**Question**: *What are the two features that are similar?*

#### Purpose
Identifies highly correlated features (redundancy) by calculating Pearson correlation coefficients between all feature pairs.

#### Method
- Calculates **Pearson correlation coefficient** between all feature pairs
- Identifies the pair with highest absolute correlation
- Displays the most correlated pair

**Pearson Correlation Formula**:
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]

where:
- r ∈ [-1, 1]
- r = 1: perfect positive correlation
- r = -1: perfect negative correlation
- r = 0: no correlation
```

#### Results & Analysis

![Most Similar Features](readme_images/most_similar_features.png)

**Answer: Astronomy and Defense Against the Dark Arts**

**Reasoning:**
The scatter plot reveals a **nearly perfect linear relationship** between these two features:
- Points form a clear diagonal line
- All four houses follow the same linear trend
- This indicates very high positive correlation (r ≈ 0.99)

**Implication for feature selection:**
Since these features are highly correlated, they provide **redundant information**. For logistic regression:
- Using both would cause **multicollinearity** issues
- The model becomes unstable and harder to interpret
- **Decision**: Keep only ONE of these features (either Astronomy OR Defense Against the Dark Arts)

---

### Pair Plot Analysis

**Question**: *From this visualization, which features are you going to use for your logistic regression?*

#### Purpose
A pair plot (scatter plot matrix) provides a comprehensive overview of all feature relationships simultaneously, helping identify:
- Features with good class separation
- Redundant/correlated features
- Distribution characteristics

#### Method
- Creates a matrix of scatter plots for all feature combinations
- **Diagonal**: Histograms showing individual feature distributions
- **Off-diagonal**: Scatter plots showing relationships between feature pairs
- Color-coded by house

#### Visual Analysis

![Pair Plot Matrix](readme_images/pair_plot_matrix.png)

#### Feature Selection Criteria

We analyze each feature based on:

1. **Class Separation** (Primary criterion)
   - ✅ Good: Distinct clusters for each house with minimal overlap
   - ❌ Poor: All houses mixed together

2. **Distribution Characteristics** (Secondary criterion)
   - Look at diagonal histograms
   - Different distributions per house = discriminative power

3. **Avoid Redundancy**
   - Eliminate highly correlated features

#### Feature-by-Feature Analysis

| Feature | Separation Quality | Decision | Reasoning |
|---------|-------------------|----------|-----------|
| **Astronomy** | ⭐⭐⭐⭐⭐ Excellent | ✅ KEEP | Clear vertical/horizontal separation, distinct clusters |
| **Herbology** | ⭐⭐⭐⭐⭐ Excellent | ✅ KEEP | Very good separation, especially Gryffindor/Slytherin |
| **Defense Against the Dark Arts** | ⭐⭐⭐⭐ Very Good | ❌ REMOVE | High correlation with Astronomy (redundant) |
| **Ancient Runes** | ⭐⭐⭐⭐ Very Good | ✅ KEEP | Good horizontal separation between houses |
| **Divination** | ⭐⭐⭐⭐ Very Good | ✅ KEEP | Clear clusters, good separation |
| **Charms** | ⭐⭐⭐ Good | ✅ KEEP | Hufflepuff well separated, helps distinguish classes |
| **Flying** | ⭐⭐⭐ Good | ✅ KEEP | Gryffindor distinct, adds complementary information |
| **History of Magic** | ⭐⭐⭐ Moderate | ⚠️ OPTIONAL | Some separation but significant overlap |
| **Transfiguration** | ⭐⭐ Poor | ❌ REMOVE | High overlap between houses |
| **Arithmancy** | ⭐ Very Poor | ❌ REMOVE | Complete mixing of all houses, no discriminative power |
| **Care of Magical Creatures** | ⭐⭐ Poor | ❌ REMOVE | Homogeneous distributions (confirmed by histogram) |
| **Muggle Studies** | ⭐⭐ Poor | ❌ REMOVE | Too much overlap |
| **Potions** | ⭐⭐ Poor | ❌ REMOVE | Mixed classes |

#### Final Feature Selection

**Selected Features for Logistic Regression:**

1. ✅ **Astronomy** - Excellent separator
2. ✅ **Herbology** - Excellent separator  
3. ✅ **Ancient Runes** - Very good separator
4. ✅ **Divination** - Very good separator
5. ✅ **Charms** - Good separator
6. ✅ **Flying** - Good separator, complements other features

**Total: 6 features**

**Excluded Features:**
- ❌ **Defense Against the Dark Arts** - Redundant with Astronomy (high correlation)
- ❌ **Arithmancy** - No discriminative power
- ❌ **Care of Magical Creatures** - Homogeneous distribution
- ❌ **Transfiguration, Muggle Studies, Potions** - Poor separation
- ⚠️ **History of Magic** - Borderline, excluded to keep model parsimonious

#### Why This Selection Works

**Strengths:**
- Each selected feature shows clear visual separation between at least 2-3 houses
- No redundant features (removed Defense Against the Dark Arts)
- Diverse separation patterns provide complementary information
- Balance between model complexity and performance

**Expected Performance:**
With these 6 features, we should achieve >98% accuracy because:
- Each house has unique "signature" across multiple features
- Overlaps in one feature are compensated by clear separation in others
- The combination provides sufficient information for reliable classification

---

## Usage

### Data Analysis
```bash
python src/describe.py
```

### Data Visualization
```bash
# Question 1: Homogeneous distribution
python src/histogram.py

# Question 2: Most similar features  
python src/scatter_plot.py

# Question 3: Feature selection for logistic regression
python src/pair_plot.py
```