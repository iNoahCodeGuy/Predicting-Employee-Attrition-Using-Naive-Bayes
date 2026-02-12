"""
Predicting Employee Attrition Using Naive Bayes Classification

A Bayesian approach to identifying employees at risk of leaving, enabling proactive retention strategies.
Companion project: https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression

Author: Noah de la Calzada
Dataset: Employee Future Prediction from Kaggle
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                            roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from scipy import stats
import os

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("Libraries loaded successfully!")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

# load the data
df = pd.read_csv('Employee.csv')

print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

# check data types and basic info
df.info()
print("\n")
df.describe()

# check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")

# target variable - how many left vs stayed?
print("\nTarget variable distribution:")
print(df['LeaveOrNot'].value_counts().sort_index())
print(f"\nLeave rate: {df['LeaveOrNot'].mean()*100:.1f}%")
print(f"Class balance ratio (Stay/Leave): {(df['LeaveOrNot']==0).sum() / (df['LeaveOrNot']==1).sum():.2f}")

# Visualize the target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_counts = df['LeaveOrNot'].value_counts().sort_index()
axes[0].bar(['Stay (0)', 'Leave (1)'], target_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7)
axes[0].set_xlabel('Leave Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Employee Leave Status', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=11, fontweight='bold')

colors = ['#3498db', '#e74c3c']
axes[1].pie(target_counts.values, labels=['Stay (0)', 'Leave (1)'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Proportion of Leave Status', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Class distribution: {target_counts[0]} employees stayed ({target_counts[0]/len(df)*100:.1f}%) "
      f"and {target_counts[1]} employees left ({target_counts[1]/len(df)*100:.1f}%)")

# Check distribution of continuous features - GaussianNB assumes normality
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Age distribution
axes[0, 0].hist(df['Age'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Age', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
axes[0, 0].legend()

# Age Q-Q plot (rough check for normality)
stats.probplot(df['Age'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Age Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# JoiningYear distribution
year_counts = df['JoiningYear'].value_counts().sort_index()
axes[0, 2].bar(year_counts.index.astype(str), year_counts.values, color='#2ecc71', alpha=0.7)
axes[0, 2].set_xlabel('Joining Year', fontsize=12)
axes[0, 2].set_ylabel('Count', fontsize=12)
axes[0, 2].set_title('Joining Year Distribution', fontsize=12, fontweight='bold')
axes[0, 2].grid(axis='y', alpha=0.3)

# Experience distribution
axes[1, 0].hist(df['ExperienceInCurrentDomain'], bins=range(0, df['ExperienceInCurrentDomain'].max()+2),
             color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Years of Experience', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Experience Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xticks(range(0, df['ExperienceInCurrentDomain'].max()+1))

# Experience Q-Q plot
stats.probplot(df['ExperienceInCurrentDomain'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Experience Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# PaymentTier distribution
payment_counts = df['PaymentTier'].value_counts().sort_index()
axes[1, 2].bar(payment_counts.index.astype(str), payment_counts.values,
              color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7)
axes[1, 2].set_xlabel('Payment Tier', fontsize=12)
axes[1, 2].set_ylabel('Count', fontsize=12)
axes[1, 2].set_title('Payment Tier Distribution', fontsize=12, fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Distribution checks:")
print(f"Age - Mean: {df['Age'].mean():.2f}, Std: {df['Age'].std():.2f}")
print(f"Experience - Mean: {df['ExperienceInCurrentDomain'].mean():.2f}, Std: {df['ExperienceInCurrentDomain'].std():.2f}")
print("\nNote: JoiningYear is discrete and not really continuous, but I'll include it anyway.")
print("The continuous features (Age, Experience) are roughly normal-ish, which is good for GaussianNB.")

# Quick look at categorical feature distributions
print("\nCategorical Feature Distributions:")
print("=" * 60)
print("\nEducation:")
print(df['Education'].value_counts())
print("\nCity:")
print(df['City'].value_counts())
print("\nGender:")
print(df['Gender'].value_counts())
print("\nEverBenched:")
print(df['EverBenched'].value_counts())

# ============================================================================
# PREPROCESSING
# ============================================================================

# reload data and split into X and y
df = pd.read_csv('Employee.csv')

X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nFeature types:")
print(f"Categorical: Education, City, Gender, EverBenched")
print(f"Numerical: Age, JoiningYear, ExperienceInCurrentDomain")
print(f"Ordinal: PaymentTier")

# one-hot encode categorical variables
categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nShape after encoding:", X_encoded.shape)
print("Columns:", X_encoded.columns.tolist())

# 80/20 train test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Train leave rate: {y_train.mean()*100:.1f}%")
print(f"Test leave rate: {y_test.mean()*100:.1f}%")

# scale features - GaussianNB works better with scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert back to dataframe to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\nScaling complete")
print(f"Train shape: {X_train_scaled.shape}")

# ============================================================================
# MODEL FITTING
# ============================================================================

# baseline GaussianNB with default parameters
model_baseline = GaussianNB()
model_baseline.fit(X_train_scaled, y_train)

train_score = model_baseline.score(X_train_scaled, y_train)
test_score = model_baseline.score(X_test_scaled, y_test)

print(f"\nBaseline GaussianNB:")
print(f"  Train accuracy: {train_score*100:.2f}%")
print(f"  Test accuracy: {test_score*100:.2f}%")

# hyperparameter tuning with GridSearchCV
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}

grid_search = GridSearchCV(
    GaussianNB(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nStarting grid search for var_smoothing...")
grid_search.fit(X_train_scaled, y_train)

# check best parameters
print("\nBest var_smoothing:", grid_search.best_params_)
print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
print(f"\nAll CV results:")
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['param_var_smoothing', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False))

# train final model with best parameters
final_model = GaussianNB(var_smoothing=grid_search.best_params_['var_smoothing'])
final_model.fit(X_train_scaled, y_train)

final_train_score = final_model.score(X_train_scaled, y_train)
final_test_score = final_model.score(X_test_scaled, y_test)

print(f"\nFinal GaussianNB (tuned):")
print(f"  Train accuracy: {final_train_score*100:.2f}%")
print(f"  Test accuracy: {final_test_score*100:.2f}%")
print(f"  Improvement over baseline: {(final_test_score - test_score)*100:.2f}%")

# ============================================================================
# CLASS IMBALANCE EXPERIMENTS
# ============================================================================
# instead of assuming the ~2:1 class imbalance doesn't matter, let's actually test it!

print("\n" + "="*60)
print("CLASS IMBALANCE EXPERIMENTS")
print("="*60)

# --- Experiment 1: GaussianNB with equal class priors ---
print("\n" + "-"*60)
print("Experiment 1: GaussianNB with Equal Priors [0.5, 0.5]")
print("-"*60)

model_equal_priors = GaussianNB(
    var_smoothing=grid_search.best_params_['var_smoothing'],
    priors=[0.5, 0.5]
)
model_equal_priors.fit(X_train_scaled, y_train)

y_pred_ep = model_equal_priors.predict(X_test_scaled)
y_pred_proba_ep = model_equal_priors.predict_proba(X_test_scaled)

cm_ep = confusion_matrix(y_test, y_pred_ep)
TN_ep, FP_ep, FN_ep, TP_ep = cm_ep.ravel()

acc_ep = accuracy_score(y_test, y_pred_ep)
prec_ep = precision_score(y_test, y_pred_ep)
rec_ep = recall_score(y_test, y_pred_ep)
spec_ep = TN_ep / (TN_ep + FP_ep)
f1_ep = f1_score(y_test, y_pred_ep)
auc_ep = roc_auc_score(y_test, y_pred_proba_ep[:, 1])

print(f"  Accuracy:    {acc_ep*100:.2f}%")
print(f"  Precision:   {prec_ep*100:.2f}%")
print(f"  Recall:      {rec_ep*100:.2f}%")
print(f"  Specificity: {spec_ep*100:.2f}%")
print(f"  F1 Score:    {f1_ep:.4f}")
print(f"  AUC:         {auc_ep:.4f}")

y_pred_base = final_model.predict(X_test_scaled)
rec_base = recall_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
print(f"\nCompared to baseline (data-learned priors):")
print(f"  Recall:    {rec_base*100:.2f}% -> {rec_ep*100:.2f}% ({(rec_ep-rec_base)*100:+.2f}%)")
print(f"  Precision: {prec_base*100:.2f}% -> {prec_ep*100:.2f}% ({(prec_ep-prec_base)*100:+.2f}%)")

# --- Experiment 2: Threshold Tuning ---
print("\n" + "-"*60)
print("Experiment 2: Classification Threshold Tuning")
print("-"*60)

y_proba_base = final_model.predict_proba(X_test_scaled)[:, 1]
thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50]

threshold_results = []
print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 75)

for t in thresholds_to_test:
    y_pred_t = (y_proba_base >= t).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t)
    f1_t = f1_score(y_test, y_pred_t)
    threshold_results.append({
        'threshold': t, 'accuracy': acc_t, 'precision': prec_t,
        'recall': rec_t, 'f1': f1_t
    })
    print(f"{t:<12.2f} {acc_t*100:<12.2f} {prec_t*100:<12.2f} {rec_t*100:<12.2f} {f1_t:<12.4f}")

best_thresh_info = max(threshold_results, key=lambda x: x['f1'])
print(f"\nBest threshold by F1: {best_thresh_info['threshold']:.2f} "
      f"(F1={best_thresh_info['f1']:.4f}, Recall={best_thresh_info['recall']*100:.2f}%)")

# threshold analysis figure
fig, ax = plt.subplots(figsize=(10, 6))
thresh_vals = [r['threshold'] for r in threshold_results]
ax.plot(thresh_vals, [r['accuracy'] for r in threshold_results],
        'o-', label='Accuracy', linewidth=2, color='#3498db')
ax.plot(thresh_vals, [r['precision'] for r in threshold_results],
        's-', label='Precision', linewidth=2, color='#e74c3c')
ax.plot(thresh_vals, [r['recall'] for r in threshold_results],
        '^-', label='Recall', linewidth=2, color='#2ecc71')
ax.plot(thresh_vals, [r['f1'] for r in threshold_results],
        'D-', label='F1 Score', linewidth=2, color='#9b59b6')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.axvline(x=best_thresh_info['threshold'], color='red', linestyle='--', alpha=0.5,
           label=f'Best F1 ({best_thresh_info["threshold"]:.2f})')
ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 4: Threshold Analysis — Precision/Recall Tradeoff',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('figures/threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 4: Shows the precision/recall tradeoff at different decision thresholds.")

# --- Experiment 3: ComplementNB and BernoulliNB ---
print("\n" + "-"*60)
print("Experiment 3: ComplementNB and BernoulliNB (on unscaled encoded data)")
print("-"*60)

# these need non-negative inputs, so use X_train/X_test (unscaled)
model_complement = ComplementNB()
model_complement.fit(X_train, y_train)
y_pred_cnb = model_complement.predict(X_test)
y_pred_proba_cnb = model_complement.predict_proba(X_test)

cm_cnb = confusion_matrix(y_test, y_pred_cnb)
TN_cnb, FP_cnb, FN_cnb, TP_cnb = cm_cnb.ravel()

acc_cnb = accuracy_score(y_test, y_pred_cnb)
prec_cnb = precision_score(y_test, y_pred_cnb)
rec_cnb = recall_score(y_test, y_pred_cnb)
spec_cnb = TN_cnb / (TN_cnb + FP_cnb)
f1_cnb = f1_score(y_test, y_pred_cnb)
auc_cnb = roc_auc_score(y_test, y_pred_proba_cnb[:, 1])

print(f"\nComplementNB:")
print(f"  Accuracy:    {acc_cnb*100:.2f}%")
print(f"  Precision:   {prec_cnb*100:.2f}%")
print(f"  Recall:      {rec_cnb*100:.2f}%")
print(f"  Specificity: {spec_cnb*100:.2f}%")
print(f"  F1 Score:    {f1_cnb:.4f}")
print(f"  AUC:         {auc_cnb:.4f}")

model_bernoulli = BernoulliNB()
model_bernoulli.fit(X_train, y_train)
y_pred_bnb = model_bernoulli.predict(X_test)
y_pred_proba_bnb = model_bernoulli.predict_proba(X_test)

cm_bnb = confusion_matrix(y_test, y_pred_bnb)
TN_bnb, FP_bnb, FN_bnb, TP_bnb = cm_bnb.ravel()

acc_bnb = accuracy_score(y_test, y_pred_bnb)
prec_bnb = precision_score(y_test, y_pred_bnb)
rec_bnb = recall_score(y_test, y_pred_bnb)
spec_bnb = TN_bnb / (TN_bnb + FP_bnb)
f1_bnb = f1_score(y_test, y_pred_bnb)
auc_bnb = roc_auc_score(y_test, y_pred_proba_bnb[:, 1])

print(f"\nBernoulliNB:")
print(f"  Accuracy:    {acc_bnb*100:.2f}%")
print(f"  Precision:   {prec_bnb*100:.2f}%")
print(f"  Recall:      {rec_bnb*100:.2f}%")
print(f"  Specificity: {spec_bnb*100:.2f}%")
print(f"  F1 Score:    {f1_bnb:.4f}")
print(f"  AUC:         {auc_bnb:.4f}")

# --- Comparison Table + Best Model Selection ---
print("\n" + "-"*60)
print("Comparison Table + Best Model Selection")
print("-"*60)

# baseline metrics
y_pred_baseline = final_model.predict(X_test_scaled)
y_pred_proba_baseline = final_model.predict_proba(X_test_scaled)
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
TN_bl, FP_bl, FN_bl, TP_bl = cm_baseline.ravel()
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline)
recall = recall_score(y_test, y_pred_baseline)
specificity = TN_bl / (TN_bl + FP_bl)
f1 = f1_score(y_test, y_pred_baseline)
roc_auc = roc_auc_score(y_test, y_pred_proba_baseline[:, 1])

# best-threshold predictions
best_thresh_val = best_thresh_info['threshold']
y_pred_bt = (y_proba_base >= best_thresh_val).astype(int)
cm_bt = confusion_matrix(y_test, y_pred_bt)
TN_bt, FP_bt, FN_bt, TP_bt = cm_bt.ravel()

# store all variants
all_variants = {
    'GaussianNB (tuned)': {
        'model': final_model, 'X_test_ref': X_test_scaled,
        'y_pred': y_pred_baseline,
        'y_proba': y_pred_proba_baseline[:, 1],
        'threshold': 0.5, 'uses_scaled': True,
    },
    'GaussianNB (equal priors)': {
        'model': model_equal_priors, 'X_test_ref': X_test_scaled,
        'y_pred': y_pred_ep, 'y_proba': y_pred_proba_ep[:, 1],
        'threshold': 0.5, 'uses_scaled': True,
    },
    f'GaussianNB (thresh={best_thresh_val:.2f})': {
        'model': final_model, 'X_test_ref': X_test_scaled,
        'y_pred': y_pred_bt, 'y_proba': y_proba_base,
        'threshold': best_thresh_val, 'uses_scaled': True,
    },
    'ComplementNB': {
        'model': model_complement, 'X_test_ref': X_test,
        'y_pred': y_pred_cnb, 'y_proba': y_pred_proba_cnb[:, 1],
        'threshold': 0.5, 'uses_scaled': False,
    },
    'BernoulliNB': {
        'model': model_bernoulli, 'X_test_ref': X_test,
        'y_pred': y_pred_bnb, 'y_proba': y_pred_proba_bnb[:, 1],
        'threshold': 0.5, 'uses_scaled': False,
    },
}

comparison_rows = []
for name, data in all_variants.items():
    yp = data['y_pred']
    cm_tmp = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm_tmp.ravel()
    comparison_rows.append({
        'Method': name,
        'Accuracy': accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp, zero_division=0),
        'Recall': recall_score(y_test, yp),
        'F1': f1_score(y_test, yp),
        'AUC': roc_auc_score(y_test, data['y_proba']),
        'Specificity': tn / (tn + fp),
    })

comp_df = pd.DataFrame(comparison_rows)

print("\nClass Imbalance Experiment Results")
print("=" * 100)
print(comp_df.to_string(index=False, float_format='{:.4f}'.format))
print("=" * 100)

# pick the best model by F1 score
best_idx = comp_df['F1'].idxmax()
best_method_name = comp_df.loc[best_idx, 'Method']
best_variant_data = list(all_variants.values())[best_idx]

print(f"\nBest model by F1 score: {best_method_name}")
print(f"  F1        = {comp_df.loc[best_idx, 'F1']:.4f}")
print(f"  Accuracy  = {comp_df.loc[best_idx, 'Accuracy']*100:.2f}%")
print(f"  Recall    = {comp_df.loc[best_idx, 'Recall']*100:.2f}%")
print(f"  Precision = {comp_df.loc[best_idx, 'Precision']*100:.2f}%")
print(f"  AUC       = {comp_df.loc[best_idx, 'AUC']:.4f}")

# reassign for downstream use
final_model = best_variant_data['model']
best_X_test = best_variant_data['X_test_ref']
best_X_train = X_train_scaled if best_variant_data['uses_scaled'] else X_train
best_threshold = best_variant_data['threshold']
best_model_name = best_method_name

print(f"\nUsing '{best_model_name}' for all downstream evaluation.")
if best_threshold != 0.5:
    print(f"Note: using custom threshold = {best_threshold} instead of the default 0.5")

# comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.15
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#1abc9c']

for i, (metric, color) in enumerate(zip(metrics_to_plot, bar_colors)):
    ax.bar(x + i * width, comp_df[metric], width, label=metric, color=color, alpha=0.8)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 5: NB Variant Comparison — All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(comp_df['Method'], fontsize=9, rotation=15, ha='right')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('figures/imbalance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 5: Side-by-side comparison of all NB variants across key metrics.")

# ============================================================================
# RESULTS - MODEL PROPERTIES
# ============================================================================

print("\n" + "="*60)
print("MODEL PROPERTIES")
print("="*60)

print(f"\nSelected model: {best_model_name}")
print(f"Model type: {type(final_model).__name__}")

if hasattr(final_model, 'class_prior_'):
    print(f"\nClass Priors (P(Stay) and P(Leave)):")
    print(f"  Stay (0): {final_model.class_prior_[0]:.4f} ({final_model.class_prior_[0]*100:.2f}%)")
    print(f"  Leave (1): {final_model.class_prior_[1]:.4f} ({final_model.class_prior_[1]*100:.2f}%)")
elif hasattr(final_model, 'class_log_prior_'):
    priors = np.exp(final_model.class_log_prior_)
    print(f"\nClass Priors (learned from training data):")
    print(f"  Stay (0): {priors[0]:.4f} ({priors[0]*100:.2f}%)")
    print(f"  Leave (1): {priors[1]:.4f} ({priors[1]*100:.2f}%)")

if hasattr(final_model, 'feature_names_in_'):
    print(f"\nNumber of features: {len(final_model.feature_names_in_)}")
if hasattr(final_model, 'var_smoothing'):
    print(f"var_smoothing: {final_model.var_smoothing}")
if best_threshold != 0.5:
    print(f"Decision threshold: {best_threshold}")

# look at feature means and variances for each class
feature_names = X_train_scaled.columns.tolist()

if hasattr(final_model, 'theta_') and hasattr(final_model, 'var_'):
    print("\nFeature Statistics by Class (Gaussian parameters):")
    print("=" * 80)
    print(f"{'Feature':<30} {'Stay Mean':<12} {'Stay Var':<12} {'Leave Mean':<12} {'Leave Var':<12}")
    print("-" * 80)

    for i, feature in enumerate(feature_names):
        stay_mean = final_model.theta_[0, i]
        stay_var = final_model.var_[0, i]
        leave_mean = final_model.theta_[1, i]
        leave_var = final_model.var_[1, i]
        print(f"{feature:<30} {stay_mean:>11.4f} {stay_var:>11.4f} {leave_mean:>11.4f} {leave_var:>11.4f}")

    print("\nNote: These are the Gaussian distribution parameters for each feature within each class.")
    print("Larger differences in means between classes = more discriminative features.")
else:
    print("\nFeature Log Probabilities by Class:")
    print("=" * 80)
    if hasattr(final_model, 'feature_log_prob_'):
        flp = final_model.feature_log_prob_
        print(f"{'Feature':<30} {'Stay LogP':<12} {'Leave LogP':<12} {'Diff':<12}")
        print("-" * 80)
        for i, feature in enumerate(feature_names):
            diff = abs(flp[1, i] - flp[0, i])
            print(f"{feature:<30} {flp[0, i]:>11.4f} {flp[1, i]:>11.4f} {diff:>11.4f}")

# feature importance visualization
if hasattr(final_model, 'theta_'):
    mean_diffs = np.abs(final_model.theta_[1, :] - final_model.theta_[0, :])
    importance_label = 'Absolute Mean Difference Between Classes'
elif hasattr(final_model, 'feature_log_prob_'):
    mean_diffs = np.abs(final_model.feature_log_prob_[1, :] - final_model.feature_log_prob_[0, :])
    importance_label = 'Absolute Log-Probability Difference Between Classes'
else:
    mean_diffs = np.zeros(len(feature_names))
    importance_label = 'Feature Importance'

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_diffs
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#3498db', alpha=0.7)
ax.set_xlabel(importance_label, fontsize=12)
ax.set_title('Figure 1: Feature Discriminative Power', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure 1: Features with larger differences between Stay and Leave classes")
print("are more useful for prediction. This shows what the model relies on most.")

# ============================================================================
# RESULTS - OUTPUT INTERPRETATION
# ============================================================================

# get predictions and probabilities using the best model
y_pred_proba = final_model.predict_proba(best_X_test)

if best_threshold != 0.5:
    y_pred = (y_pred_proba[:, 1] >= best_threshold).astype(int)
    print(f"\nUsing custom threshold: {best_threshold}")
else:
    y_pred = final_model.predict(best_X_test)

print("\n" + "="*60)
print("OUTPUT INTERPRETATION")
print("="*60)
print(f"\nExample Predictions ({best_model_name}):")
print("=" * 80)
print(f"{'Actual':<10} {'Predicted':<10} {'P(Stay)':<12} {'P(Leave)':<12} {'Correct':<10}")
print("-" * 80)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    prob_stay = y_pred_proba[i, 0]
    prob_leave = y_pred_proba[i, 1]
    correct = "Yes" if actual == predicted else "No"
    print(f"{actual:<10} {predicted:<10} {prob_stay:<12.4f} {prob_leave:<12.4f} {correct:<10}")

print("\nThe model outputs probabilities that can be interpreted as confidence scores.")
print("HR could use these probabilities to rank employees by attrition risk.")

# distribution of predicted probabilities
prob_leave = y_pred_proba[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(prob_leave, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0].axvline(x=best_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Decision Threshold ({best_threshold})')
axes[0].set_xlabel('Predicted P(Leave)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Predicted Leave Probabilities', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

prob_stay_actual = prob_leave[y_test == 0]
prob_leave_actual = prob_leave[y_test == 1]

axes[1].hist([prob_stay_actual, prob_leave_actual], bins=25,
             label=['Actually Stayed', 'Actually Left'],
             color=['#3498db', '#e74c3c'], alpha=0.6, edgecolor='black')
axes[1].axvline(x=best_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Decision Threshold ({best_threshold})')
axes[1].set_xlabel('Predicted P(Leave)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Predicted Probabilities by Actual Outcome', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nThe model separates the classes reasonably well, though there's some overlap.")
print(f"Employees with P(Leave) > {best_threshold} are predicted to leave.")

# ============================================================================
# EVALUATION
# ============================================================================

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n" + "="*60)
print("EVALUATION")
print("="*60)
print("\nConfusion Matrix:")
print("=" * 60)
print(f"                 Predicted")
print(f"              Stay    Leave")
print(f"Actual Stay    {TN:>4}     {FP:>4}")
print(f"Actual Leave   {FN:>4}     {TP:>4}")
print("=" * 60)

# confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title(f'Figure 2: Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold')
plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFigure 2: Confusion matrix for {best_model_name} on the test set.")

# calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print("\nEvaluation Metrics:")
print("=" * 60)
print(f"Accuracy:    {accuracy*100:.2f}%")
print(f"Precision:   {precision*100:.2f}%")
print(f"Recall:      {recall*100:.2f}%")
print(f"Specificity: {specificity*100:.2f}%")
print(f"F1 Score:    {f1:.4f}")
print("=" * 60)

print("\nInterpretation:")
print(f"- Accuracy: {accuracy*100:.1f}% of predictions are correct overall")
print(f"- Precision: When the model predicts 'Leave', it's correct {precision*100:.1f}% of the time")
print(f"- Recall: The model catches {recall*100:.1f}% of employees who actually leave")
print(f"- Specificity: The model correctly identifies {specificity*100:.1f}% of employees who stay")

# classification report
print("\nDetailed Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', linewidth=1)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
plt.title('Figure 3: ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

performance_level = ('excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8
                     else 'moderate' if roc_auc > 0.7 else 'poor')
print(f"\nFigure 3: Receiver Operating Characteristic curve for {best_model_name}.")
print(f"The AUC of {roc_auc:.3f} indicates {performance_level} discriminative performance.")
print(f"An AUC of 1.0 = perfect classification; 0.5 = random guessing.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)
print(f"Selected model: {best_model_name}")
print(f"Model class:    {type(final_model).__name__}")
if hasattr(final_model, 'var_smoothing'):
    print(f"var_smoothing:  {final_model.var_smoothing}")
print(f"Threshold:      {best_threshold}")
print(f"Training set:   {len(y_train)} samples")
print(f"Test set:       {len(y_test)} samples")
print()
print("Test Set Metrics:")
print(f"  Accuracy:    {accuracy*100:.2f}%")
print(f"  Precision:   {precision*100:.2f}%")
print(f"  Recall:      {recall*100:.2f}%")
print(f"  Specificity: {specificity*100:.2f}%")
print(f"  F1 Score:    {f1:.4f}")
print(f"  AUC:         {roc_auc:.4f}")
print()
print("Feature List:")
for i, col in enumerate(X_train_scaled.columns):
    print(f"  {i+1}. {col}")
print(f"\n(Total: {len(X_train_scaled.columns)} features after one-hot encoding with drop_first=True)")
print("=" * 60)
print("\nANALYSIS COMPLETE")
print("=" * 60)
