# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DSLR (Data Science × Logistic Regression)** - A machine learning educational project implementing multi-class classification from scratch to sort Harry Potter students into their Hogwarts houses (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) based on academic performance.

**Goal**: Achieve ≥98% accuracy using one-vs-all logistic regression with gradient descent.

**Critical Constraints**:
- **FORBIDDEN**: Any function that does the job for you (count, mean, std, min, max, percentile, describe(), etc.)
- **FORBIDDEN**: sklearn or any ML library for training (only sklearn.metrics for evaluation)
- All statistical and ML algorithms must be implemented manually from scratch
- **Minimum Performance**: Must achieve ≥98% accuracy (evaluated with sklearn's accuracy_score)

## Recommended Workflow

**IMPORTANT**: The project specification strongly recommends completing tasks in this order:
1. Data Analysis (Part 1) - Implement describe.py ✅ COMPLETED
2. Data Visualization (Part 2) - Create histogram, scatter_plot, pair_plot ✅ COMPLETED
3. Logistic Regression (Part 3) - Implement logreg_train and logreg_predict ⏳ IN PROGRESS

## Common Commands

### Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Analysis (Part 1)
```bash
# Display statistical analysis for all numerical features
python src/data/describe.py datasets/dataset_train.csv
```

### Data Visualization (Part 2)
```bash
# Generate histogram analysis (identifies homogeneous features)
python src/data/histogram.py

# Generate scatter plot analysis (identifies correlated features)
python src/data/scatter_plot.py

# Generate pair plot matrix (for feature selection)
python src/data/pair_plot.py
```

**Note**: Visualizations are saved to `histograms/`, `scatter_plots/`, and `pair_plots/` directories respectively (gitignored).

### Logistic Regression (Part 3 - Pending Implementation)
```bash
# Train the model (to be implemented)
python logreg_train.py datasets/dataset_train.csv
# This generates a weights file (e.g., weights.csv)

# Make predictions (to be implemented)
python logreg_predict.py datasets/dataset_test.csv weights.csv
# This generates houses.csv with predictions

# Expected output format for houses.csv:
# Index,Hogwarts House
# 0,Gryffindor
# 1,Hufflepuff
# 2,Ravenclaw
# ...
```

## Architecture

### Modular Analysis Pipeline

The codebase follows a **class-based modular architecture** with clear separation of concerns:

```
Data Layer (CSV files in datasets/)
    ↓
Utility Layer (src/data/utils.py)
    ↓  ← Shared functions: get_min/max, get_numeric_features, etc.
    ↓
Analysis Layer (src/data/*.py classes)
    ├── Describe: Statistical analysis (17 metrics)
    ├── Histogram: Distribution homogeneity analysis
    ├── ScatterPlot: Correlation analysis (Pearson coefficient)
    └── PairPlot: Feature selection via scatter plot matrix
    ↓
Visualization Layer (matplotlib)
    ↓
Output (PNG files + console statistics)
```

### Key Design Patterns

1. **Manual Statistical Implementation**: All statistics in `Describe` class are implemented from first principles (mean, std, percentiles, skewness, kurtosis, etc.) without using NumPy statistical functions.

2. **Consistent Color Coding**: All visualizations use the same house colors:
   - Gryffindor = Red
   - Hufflepuff = Yellow
   - Ravenclaw = Blue
   - Slytherin = Green

3. **Shared Utilities**: Common operations (min/max, numeric feature extraction, directory creation) are centralized in `src/data/utils.py` to maintain DRY principles.

### Critical Implementation Details

#### Statistical Calculations (src/data/describe.py)

- **Standard Deviation**: Uses Bessel's correction (n-1) for sample std
- **Percentiles**: Linear interpolation method for non-integer index positions
  ```
  index = (percentile/100) × (n-1)
  value = lower + weight × (upper - lower)
  ```
- **Skewness**: Measures distribution asymmetry using third standardized moment
- **Kurtosis**: Measures "tailedness" using fourth standardized moment minus 3 (excess kurtosis)
- **Outlier Detection**: IQR method with 1.5×IQR threshold

#### Correlation Analysis (src/data/scatter_plot.py)

- **Pearson Correlation**: Manual implementation of correlation coefficient
  ```
  r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
  ```
- Identifies **Astronomy** and **Defense Against the Dark Arts** as highly correlated (r ≈ 0.99)

#### Feature Selection Results (from pair_plot.py analysis)

**Selected Features** (6 total):
1. Astronomy - Excellent class separation
2. Herbology - Excellent class separation
3. Ancient Runes - Very good separation
4. Divination - Very good separation
5. Charms - Good separation
6. Flying - Good separation, complements other features

**Excluded Features**:
- Defense Against the Dark Arts (redundant with Astronomy - multicollinearity)
- Care of Magical Creatures (homogeneous distribution across houses)
- Arithmancy (no discriminative power)
- Transfiguration, Muggle Studies, Potions, History of Magic (poor separation)

### Logistic Regression Implementation (Pending)

The `src/logistic_regression/` directory is currently empty and awaits implementation. The model should:

1. **Feature Preprocessing**:
   - Use only the 6 selected features listed above
   - Handle missing values (drop rows with NaN)
   - Normalize/standardize features for gradient descent convergence

2. **One-vs-All Strategy**:
   - Train 4 binary classifiers (one per house)
   - Each classifier predicts probability of one house vs. all others
   - Final prediction: argmax of the 4 probability outputs

3. **Gradient Descent** (from project specification):

   **Sigmoid (Logistic) Function**:
   ```
   g(z) = 1 / (1 + e^(-z))
   ```

   **Hypothesis Function**:
   ```
   hθ(x) = g(θ^T * x) = 1 / (1 + e^(-θ^T * x))
   ```

   **Cost (Loss) Function** (Binary Cross-Entropy):
   ```
   J(θ) = -(1/m) * Σ[y^i * log(hθ(x^i)) + (1 - y^i) * log(1 - hθ(x^i))]
   ```
   where m = number of training examples

   **Gradient (Partial Derivative)**:
   ```
   ∂J(θ)/∂θj = (1/m) * Σ[(hθ(x^i) - y^i) * xj^i]
   ```

   **Update Rule**:
   ```
   θj := θj - α * ∂J(θ)/∂θj
   ```
   where α = learning rate

   - No sklearn for training - only for evaluation metrics (accuracy_score)

4. **Output Format**:
   - Save trained weights to CSV file
   - Prediction output should include student index and predicted house

## Bonus Options

The following bonus features can be implemented after completing the mandatory part:

1. **Enhanced Statistics**: Add more statistical fields to describe output (already implemented: skewness, kurtosis, MAD, outliers, etc.)
2. **Stochastic Gradient Descent**: Update weights after each training example instead of using batch gradient descent
3. **Alternative Optimization Algorithms**:
   - Batch Gradient Descent (standard approach)
   - Mini-batch Gradient Descent (compromise between batch and stochastic)
   - Advanced optimizers (Adam, RMSprop, etc.)

**Note**: Bonus features are only evaluated if the mandatory part achieves perfect functionality and ≥98% accuracy.

## Development Notes

### Project Status
- **Current Branch**: `feat(logistic_regression)`
- **Completed**: Data analysis (Part 1) and visualization (Part 2)
- **In Progress**: Logistic regression implementation (Part 3)

### Important Files
- `datasets/dataset_train.csv`: Training data (1600 students, 13 courses + house labels)
- `datasets/dataset_test.csv`: Test data (400 students, no house labels)
- `src/data/utils.py`: Shared utility functions used across all analysis modules
- `README.md`: Comprehensive documentation with analysis results and visualizations

### Evaluation Criteria

**From Project Specification**:
1. **No Heavy-Lifting Functions**: Code will be checked to ensure no built-in statistical or ML functions were used
2. **Accuracy Requirement**: Predictions on `dataset_test.csv` must achieve ≥98% accuracy (measured with sklearn.metrics.accuracy_score)
3. **Understanding**: Must be able to explain how logistic regression, gradient descent, and one-vs-all strategy work
4. **Correct File Names**: Programs must be named exactly as specified (describe, histogram, scatter_plot, pair_plot, logreg_train, logreg_predict)
5. **Output Format**: houses.csv must match the exact format specified (Index,Hogwarts House columns)

### Testing Approach
- Manually verify statistical outputs against pandas `describe()` for correctness
- Visual inspection of plots to ensure proper house separation
- Test logistic regression on training data first, then on test data
- Compare predictions against expected ≥98% accuracy threshold
- Only use sklearn for final model evaluation, not for training
- Be prepared to explain the mathematics and algorithmic choices during peer evaluation

## References

See [README.md](README.md) for detailed explanations of:
- Statistical formulas and their implementations
- Visual analysis of each feature's discriminative power
- Reasoning behind feature selection decisions
- Expected model performance with selected features
