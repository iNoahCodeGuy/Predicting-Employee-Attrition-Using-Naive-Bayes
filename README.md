# Predicting Employee Attrition Using Naive Bayes Classification

Employee attrition is fundamentally a prediction problem with asymmetric costs: failing to identify an at-risk employee means losing institutional knowledge, incurring recruitment expenses, and disrupting team productivity. Flagging someone who stays is far less costly. This asymmetry makes the choice of model — and its precision/recall characteristics — a practical decision, not just an academic one.

This project applies **Naive Bayes classification** to predict employee attrition, using Bayes' theorem to compute posterior probabilities directly from the data. It serves as a companion to a [prior logistic regression analysis](https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression) on the same dataset, making it possible to compare how two fundamentally different modeling assumptions — generative (Naive Bayes) vs. discriminative (logistic regression) — handle the same prediction task.

## Model Results

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### ROC Curve

![ROC Curve](roc_curve.png)

## Key Findings

### Comparison with Logistic Regression

| Metric | Naive Bayes | Logistic Regression | Difference |
|--------|-------------|---------------------|------------|
| Accuracy | 72.07% | 75.51% | -3.44% |
| Precision | 59.62% | 71.50% | -11.88% |
| Recall | 58.13% | 47.81% | **+10.32%** |
| AUC | 0.7249 | 0.7399 | -0.015 |

The most important number in this table is recall. Naive Bayes catches 58% of employees who actually leave, compared to 48% for logistic regression — a 10 percentage point improvement. In an HR context where missing an at-risk employee is more costly than a false alarm, this tradeoff favors Naive Bayes for retention-focused strategies.

The two models make different types of errors, which suggests potential value in an ensemble approach.

### Top Discriminative Features

The Bayesian likelihood analysis identified the features with the largest distributional differences between Stay and Leave classes:

1. **Gender_Male** (0.4513)
2. **PaymentTier** (0.4102)
3. **JoiningYear** (0.4055)
4. **City_Pune** (0.3997)
5. **Education_Masters** (0.3025)

These align with patterns observed in the prior exploratory analysis, providing cross-method validation.

## Methodology

### 1. Exploratory Analysis
- Dataset: 4,653 employees, 9 features
- Target: LeaveOrNot (34.4% leave rate — a ~2:1 class imbalance)
- Distributional checks: Q-Q plots to assess GaussianNB's normality assumption

### 2. Preprocessing
- One-hot encoding for categorical variables (Education, City, Gender, EverBenched)
- StandardScaler fit on training data only (prevents information leakage)
- 80/20 stratified train-test split (random_state=42)
- Result: 10 features (4 numerical + 6 binary indicators)

### 3. Model Development
- Baseline GaussianNB with GridSearchCV for `var_smoothing` optimization
- Best parameter: var_smoothing = 0.001
- Class imbalance experiments: equal priors, threshold tuning, ComplementNB, BernoulliNB
- Model selection by F1 score across all five variants

### 4. Evaluation
- Confusion matrix, precision, recall, specificity, F1, AUC
- ROC curve analysis
- Threshold tuning to explore precision/recall tradeoff
- Direct comparison with logistic regression baseline

## Project Structure

```
├── employee_attrition_naive_bayes.ipynb   # Main analysis notebook
├── employee_attrition_naive_bayes.py      # Standalone Python script
├── employee_attrition_naive_bayes.html    # HTML export
├── Employee.csv                           # Dataset (4,653 employees)
├── requirements.txt                       # Python dependencies
├── figures/                               # Generated visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── probability_distribution.png
│   ├── threshold_analysis.png
│   └── imbalance_comparison.png
├── confusion_matrix.png                   # Embedded in README
├── roc_curve.png                          # Embedded in README
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Naive-Bayes.git
cd Predicting-Employee-Attrition-Using-Naive-Bayes
pip install -r requirements.txt
```

### Run the Analysis

```bash
jupyter notebook employee_attrition_naive_bayes.ipynb
```

Then go to **Cell → Run All** to execute the full analysis.

Or run the standalone script:

```bash
python employee_attrition_naive_bayes.py
```

## Requirements

- `pandas>=1.5.0`
- `numpy>=1.24.0`
- `matplotlib>=3.6.0`
- `seaborn>=0.12.0`
- `scikit-learn>=1.2.0`
- `jupyter>=1.0.0`
- `scipy>=1.10.0`

## Author

**Noah de la Calzada**

[![GitHub](https://img.shields.io/badge/GitHub-iNoahCodeGuy-181717?style=flat&logo=github)](https://github.com/iNoahCodeGuy)

## License

MIT License — feel free to use this code for your own projects.
