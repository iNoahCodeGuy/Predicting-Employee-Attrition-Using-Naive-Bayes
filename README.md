# Predicting Employee Attrition Using Naive Bayes Classification

The core problem is straightforward: employees leave, and companies rarely see it coming. The cost of a missed prediction is high — recruiting, onboarding, lost institutional knowledge. The cost of a false alarm is low — you checked in on someone who was fine. That asymmetry should drive every modeling decision.

What makes attrition structurally hard to predict is that the features available to any model are proxies, not causes. We observe payment tier, tenure, location, education — but nobody leaves because of a number in a column. They leave because of management quality, competing offers, life changes, accumulated frustration. The features in any HR dataset are shadows of the real drivers, which means any classifier is working with a compressed and lossy representation of reality. The question is whether we can extract enough signal from those shadows to act on.

This project uses **Naive Bayes classification** to predict which employees are likely to leave. I built a [logistic regression model](https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression) on this same dataset previously, which gives me a direct baseline to compare against. The interesting question isn't just "does Naive Bayes work" — it's whether a generative model that learns feature distributions within each class makes different (and potentially better) mistakes than a discriminative model that draws a decision boundary directly. Logistic regression asks: "given this combination of features, which side of the boundary are you on?" Naive Bayes asks: "what does a typical leaver look like, and how much do you resemble one?" Those are fundamentally different questions, and they should produce different error profiles.

## The Dataset at a Glance

4,653 employees. 8 features. One binary target: did they leave or not?

The first thing to notice is the class imbalance. About 34% of employees left, 66% stayed. That's a ~2:1 ratio — not extreme, but enough to cause problems if you're not paying attention.

![Class Distribution](class_distribution.png)

This matters because Naive Bayes learns class priors directly from training frequencies. If 66% of your training data is "Stay," the model starts with a built-in bias toward predicting "Stay." That's fine for overall accuracy — but it kills recall on the class you actually care about.

**Dataset:** [Employee Future Prediction](https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction) (Kaggle)

## Model Results

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix tells the story concretely: 213 true positives (correctly flagged as leaving), 444 true negatives, 167 false positives, and 107 false negatives. Those 107 false negatives are the expensive ones — employees who left that the model missed. The 167 false positives are cheaper — just unnecessary check-ins.

### ROC Curve

![ROC Curve](roc_curve.png)

AUC of 0.725. Not spectacular, but meaningfully above random (0.5). The curve shows the model has real discriminative power — it's finding signal in the data, even if it's not separating the classes perfectly.

## Key Findings

### How Naive Bayes Compares to Logistic Regression

| Metric | Naive Bayes | Logistic Regression | Difference |
|--------|-------------|---------------------|------------|
| Accuracy | 72.07% | 75.51% | -3.44% |
| Precision | 59.62% | 71.50% | -11.88% |
| Recall | 58.13% | 47.81% | **+10.32%** |
| AUC | 0.7249 | 0.7399 | -0.015 |

The headline: Naive Bayes catches 10% more of the employees who actually leave. That's the recall improvement — 58% vs 48%.

The tradeoff is precision. Logistic regression is more confident when it does flag someone (72% precision vs 60%). But in an HR context, I'd rather cast a wider net and miss fewer people. A false alarm means you had a retention conversation with someone who was fine. A missed leaver means you lost them.

This answers the generative vs. discriminative question from the introduction. Naive Bayes recovers more true positives because it models what each class *looks like* independently — it builds a probability distribution for leavers and a separate one for stayers, then asks which distribution a new employee fits better. Logistic regression draws a single decision boundary through the combined feature space, optimizing for overall separation. That boundary gets pulled toward the majority class when there's imbalance. Naive Bayes is more resistant to this because its class-conditional distributions don't depend on the relative frequency of each class in the same way — the priors do, but the likelihoods don't. The result: different error profiles. Naive Bayes is more willing to flag borderline cases as leavers, which is exactly what you want when false negatives are expensive.

The two models make genuinely different types of mistakes, which isn't a flaw — it's an argument for combining them in an ensemble.

### What Features Matter Most

![Feature Importance](feature_importance.png)

The Bayesian likelihood analysis measures how different each feature's distribution is between the Stay and Leave classes. Larger differences mean the feature is more useful for prediction.

1. **Gender_Male** (0.4513) — the strongest signal. Male employees have notably different attrition patterns than female employees.
2. **PaymentTier** (0.4102) — compensation level is almost as predictive as gender.
3. **JoiningYear** (0.4055) — when someone joined tells you something about whether they'll stay.
4. **City_Pune** (0.3997) — Pune has ~50% attrition vs 27-32% elsewhere. Location matters.
5. **Education_Masters** (0.3025) — Master's holders leave at higher rates than Bachelor's or PhD holders.

These line up with what the exploratory analysis found, which is reassuring — two different analytical lenses pointing at the same features.

### The Precision/Recall Tradeoff

![Threshold Analysis](threshold_analysis.png)

This is one of the more useful outputs. The default decision threshold is 0.5 — predict "Leave" when P(Leave) > 0.5. But there's no law that says you have to use 0.5.

Lowering the threshold to 0.4 bumps recall from 58% to 67% while precision drops from 60% to 56%. Lowering it further to 0.3 gets recall up to 71%, but precision falls to 52%. The best F1 balance lands at a threshold of 0.40 (F1 = 0.6086).

In practice, this means HR can tune the sensitivity to match their budget. Have resources for more retention conversations? Lower the threshold. Need to be more targeted? Raise it.

### Class Imbalance Experiments

I didn't just try one model and call it done. The ~2:1 class imbalance motivated testing five different approaches:

![Imbalance Comparison](imbalance_comparison.png)

The comparison table from the analysis:

| Method | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|-----|-----|
| GaussianNB (tuned) | 72.07% | 59.62% | 58.13% | 0.5886 | 0.7249 |
| GaussianNB (equal priors) | 68.96% | 53.79% | 68.75% | 0.6036 | 0.7249 |
| GaussianNB (thresh=0.40) | 70.57% | 56.05% | 66.56% | 0.6086 | 0.7249 |
| ComplementNB | 68.31% | 52.96% | 70.00% | 0.6030 | 0.7316 |
| BernoulliNB | 76.58% | 72.37% | 51.56% | 0.6022 | 0.7223 |

BernoulliNB has the highest accuracy (76.58%) and precision (72.37%), but the worst recall (51.56%). It's being conservative — fewer predictions of "Leave," but more accurate when it does predict it. ComplementNB goes the other direction — highest recall (70%) but lowest precision (53%).

I selected the threshold-tuned GaussianNB (thresh=0.40) as the best model by F1 score. It hits the best balance: 66.56% recall, 56.05% precision, 0.6086 F1. The key takeaway from this whole exercise: always test instead of assume. The imbalance does matter, and different approaches handle it in meaningfully different ways.

## Methodology

### 1. Exploratory Analysis

The dataset has 4,653 employees with 8 features and a binary target (LeaveOrNot). No missing values, which is rare and nice.

The central question for EDA wasn't "which features predict attrition" — a prior exploratory analysis on this dataset had already established that (gender, city, payment tier, education). The question here was whether the data's distributional properties are compatible with the assumptions Naive Bayes makes. GaussianNB assumes continuous features are normally distributed within each class. Age checked out reasonably well. JoiningYear and ExperienceInCurrentDomain are more discrete and skewed, which means the normality assumption is only partially satisfied. Not a dealbreaker, but it sets an expectation: this model will work, but it won't work perfectly, and the imperfection has a known source.

The class imbalance (34.4% Leave vs 65.6% Stay) was the most consequential finding. It's what motivated the entire class imbalance experiment section — not as an afterthought, but as a first-order design decision.

### 2. Preprocessing

Every preprocessing choice was driven by a specific constraint of the algorithm. Four of the eight features are categorical (Education, City, Gender, EverBenched). Naive Bayes computes probability distributions over numerical inputs, so the question is how to convert categories without imposing false ordinal relationships — "Bangalore" isn't greater than "Pune."

One-hot encoding with `drop_first=True` solves this cleanly: each categorical value becomes its own binary column, and the dropped reference level avoids redundancy. This took 8 original columns to 10 features (4 numerical + 6 binary indicators). StandardScaler was fit on training data only to prevent information leakage. GaussianNB is technically scale-invariant, but scaling improves numerical stability and keeps the door open for comparing with NB variants that aren't. The stratified 80/20 split preserved the 34.4% leave rate in both partitions — 3,722 training samples, 931 test samples.

### 3. Model Development

The development followed a deliberate diagnostic sequence. Baseline GaussianNB with no tuning: 71.97% accuracy. GridSearchCV over `var_smoothing` (1e-9 to 0.001) found optimal value 0.001, yielding 72.07% — a 0.11% improvement. Basically nothing. But that negative result was informative: it told me the bottleneck isn't the smoothing parameter, it's the class imbalance distorting the priors. That diagnosis is what justified the five-variant experiment detailed in the Key Findings section above.

### 4. Evaluation

No single metric tells the full story with imbalanced classes. A model that predicts "Stay" for everyone gets 65.6% accuracy while catching zero at-risk employees — useless by the cost asymmetry established at the outset.

The evaluation framework was designed around that asymmetry:
- **Recall** is the primary metric — it measures how many actual leavers we catch, which directly determines the value of early intervention.
- **Precision** is the secondary constraint — it determines how many false alarms HR has to absorb. Acceptable as long as check-in costs remain low.
- **F1** provides a single balancing number for model selection across variants.
- **AUC** measures discriminative power across all possible thresholds, independent of any single operating point.
- **Specificity** ensures we're not drowning the majority class in false positives.

## Limitations

Honesty about what a model cannot do is as important as demonstrating what it can.

**The features are proxies, not causes.** This model identifies statistical associations between observable attributes and attrition. It cannot tell you *why* a specific employee is at risk — only that their feature profile resembles people who left. Payment tier correlates with attrition, but the causal chain (compensation dissatisfaction? role mismatch? market alternatives?) is invisible to the classifier.

**No individual-level explanations.** Naive Bayes produces a class probability, not a feature attribution. If the model flags an employee at 0.65 probability of leaving, it cannot say "mostly because of their payment tier and location." Techniques like SHAP values could add this capability but were outside the scope of this analysis.

**Temporal dynamics are absent.** The dataset is a cross-sectional snapshot. It doesn't capture trajectories — an employee whose satisfaction is declining over six months looks identical to one who's been stable. Longitudinal features would likely improve recall substantially, but they require different data infrastructure.

**The conditional independence assumption is violated.** Naive Bayes assumes all features are independent given the class. They aren't — education level and payment tier are correlated, as are joining year and experience. The model works despite this violation (a well-documented property of Naive Bayes), but it means the probability estimates are poorly calibrated even when the classifications are useful.

**Generalizability is unverified.** The dataset comes from a single company (or a specific Kaggle context). Attrition patterns are highly organization-dependent — what predicts leaving at a tech company in Pune may not predict leaving at a hospital in Chicago. Any deployment would require retraining on local data.

## Conclusion

Return to the core asymmetry: missed leavers are expensive, false alarms are cheap. Given that constraint, the threshold-tuned GaussianNB (thresh=0.40) is the right choice — it catches two-thirds of employees who actually leave while keeping false positives at a manageable level. It won't catch everyone. No model built on 8 proxy features will. But it shifts the baseline from reactive (discovering attrition after the resignation letter) to proactive (having a retention conversation while there's still time to act).

The comparison with logistic regression confirmed the theoretical expectation: a generative model and a discriminative model make different mistakes on the same data. Naive Bayes trades precision for recall — it's more willing to flag borderline cases — which aligns with the cost structure of this problem. Neither model dominates the other across all metrics, which is itself a finding worth noting. It suggests that an ensemble combining both approaches could capture the strengths of each.

The most important methodological takeaway is that default parameters are not a strategy. The baseline GaussianNB and the threshold-tuned version use the same algorithm, the same features, the same training data — but the tuned version catches 8% more leavers. That improvement came entirely from taking the class imbalance seriously and testing multiple ways to address it. The data doesn't change. The question you ask of it does.

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
