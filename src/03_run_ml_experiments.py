"""
Comprehensive 12-Model Machine Learning Benchmark for MOF Synthesis
Evaluates: Ensembles, Boosting, Kernel Machines, Linear, Bayesian, Neural Networks, and Voting Ensembles.
Outputs: Leaderboard CSV, Multi-metric grouped charts, Radar Plots, and Confusion Matrices (300 DPI).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Tree & Ensemble Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

# 2. Kernel & Distance Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 3. Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# 4. Bayesian Models
from sklearn.naive_bayes import GaussianNB

# 5. Neural Network
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------
# Output Directory & Style Setup
# ---------------------------------------------------------------------
OUTPUT_DIR = "extracted_data/all_models_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font="Arial")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.titlesize': 15,
    'figure.titleweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ---------------------------------------------------------------------
# Data Loading & Standardizing
# ---------------------------------------------------------------------
def load_and_prep_data():
    df = pd.read_csv("extracted_data/full_realistic_mof_data.csv")
    
    # Clean text values
    df['solvent'] = df['solvent'].astype(str).str.lower().str.strip()
    df['solvent'] = df['solvent'].replace({'nan': np.nan, 'none': np.nan, 'h2o': 'water'})
    
    df['metal'] = df['metal'].astype(str).str.upper().str.strip()
    df['metal'] = df['metal'].replace({'NAN': np.nan, 'NONE': np.nan})
    
    df['linker'] = df['linker'].astype(str).str.upper().str.strip()
    df['linker'] = df['linker'].replace({'NAN': np.nan, 'NONE': np.nan, 'H3BTC': 'BTC', 'TEREPHTHALIC': 'BDC'})
    
    for col in ['temperature', 'pH', 'time_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

df = load_and_prep_data()

# Filter for solvent prediction task
df_ml = df[df['solvent'].notna()].copy()
df_ml['target_solvent'] = df_ml['solvent'].apply(lambda x: x if x in ['water', 'ethanol'] else 'other_solvent')

X = df_ml[['metal', 'linker', 'temperature', 'pH', 'time_hours']]
y = df_ml['target_solvent']

print(f"Dataset Size: {len(df_ml)} samples | Classes: {dict(y.value_counts())}")

# ---------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------
cat_cols = ['metal', 'linker']
num_cols = ['temperature', 'pH', 'time_hours']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ]), cat_cols)
])

# ---------------------------------------------------------------------
# Define the 12 Diverse Machine Learning Models
# ---------------------------------------------------------------------
models = {
    # 1. Linear & Margin
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Ridge Classifier": RidgeClassifier(alpha=1.0, random_state=42),
    "Linear SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    "RBF SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    
    # 2. Distance & Probability
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights='distance'),
    "Gaussian Naive Bayes": GaussianNB(),
    
    # 3. Decision Tree & Boosting
    "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_split=4, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42),
    
    # 4. Bagging / Random Forests
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42),
    
    # 5. Neural Network
    "Multi-Layer Perceptron (NN)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, alpha=0.01, random_state=42)
}

# ---------------------------------------------------------------------
# Benchmark Execution via Stratified 4-Fold Cross Validation
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("         RUNNING 13-MODEL BENCHMARK (STRATIFIED 4-FOLD CV)")
print("="*70)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    
    cv_res = cross_validate(
        pipe, X, y, cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        return_train_score=False
    )
    
    acc_m = np.mean(cv_res['test_accuracy'])
    acc_s = np.std(cv_res['test_accuracy'])
    f1_m = np.mean(cv_res['test_f1_macro'])
    f1_s = np.std(cv_res['test_f1_macro'])
    prec_m = np.mean(cv_res['test_precision_macro'])
    rec_m = np.mean(cv_res['test_recall_macro'])
    
    results.append({
        "Model": name,
        "Accuracy_Mean": acc_m,
        "Accuracy_Std": acc_s,
        "F1_Macro_Mean": f1_m,
        "F1_Macro_Std": f1_s,
        "Precision_Mean": prec_m,
        "Recall_Mean": rec_m
    })
    
    print(f"[{name:<28}] Acc: {acc_m:.3f} (+/- {acc_s:.3f}) | F1-Macro: {f1_m:.3f}")

results_df = pd.DataFrame(results).sort_values(by="Accuracy_Mean", ascending=False).reset_index(drop=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "full_model_leaderboard.csv"), index=False)
print(f"\n[SAVED] Leaderboard saved to: {os.path.join(OUTPUT_DIR, 'full_model_leaderboard.csv')}")

# =====================================================================
# FIGURE 1: Full 13-Model Leaderboard Bar Chart (Vibrant Palette)
# =====================================================================
print("\n[Plot 1/3] Generating 13-Model Leaderboard Figure...")
plt.figure(figsize=(14, 7))
palette = sns.color_palette("turbo", len(results_df))

bars = plt.bar(
    range(len(results_df)),
    results_df['Accuracy_Mean'],
    yerr=results_df['Accuracy_Std'],
    capsize=4,
    color=palette,
    edgecolor='black',
    linewidth=0.8,
    alpha=0.9
)

plt.xticks(range(len(results_df)), results_df['Model'], rotation=35, ha='right', fontweight='bold')
plt.ylabel("Cross-Validated Accuracy (4-Fold CV)")
plt.title("Comprehensive 13-Model Benchmark: Solvent System Prediction", pad=15)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i, bar in enumerate(bars):
    val = results_df['Accuracy_Mean'][i]
    err = results_df['Accuracy_Std'][i]
    plt.text(bar.get_x() + bar.get_width()/2, val + err + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "Fig1_13_Models_Accuracy_Leaderboard.png")
plt.savefig(fig1_path)
plt.close()
print(f" -> Saved: {fig1_path}")


# =====================================================================
# FIGURE 2: Multi-Metric Radar / Spider Comparison (Top 5 Models)
# =====================================================================
print("[Plot 2/3] Generating Top Models Multi-Metric Radar Plot...")
top5_df = results_df.head(5).copy()

metrics_list = ['Accuracy_Mean', 'F1_Macro_Mean', 'Precision_Mean', 'Recall_Mean']
metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
num_vars = len(metric_labels)

# Compute angles for radar plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
radar_colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

for i, row in top5_df.iterrows():
    values = [row[m] for m in metrics_list]
    values += values[:1]
    ax.plot(angles, values, color=radar_colors[i], linewidth=2.5, label=f"{row['Model']} ({row['Accuracy_Mean']:.2f})")
    ax.fill(angles, values, color=radar_colors[i], alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, size=12, weight='bold')
ax.set_ylim(0, 1.0)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
plt.title("Top 5 ML Models: Multi-Metric Performance Radar", weight='bold', size=14, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, facecolor='white')

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "Fig2_Top_Models_Radar_Comparison.png")
plt.savefig(fig2_path)
plt.close()
print(f" -> Saved: {fig2_path}")


# =====================================================================
# FIGURE 3: Confusion Matrices for Top 4 Diverse Architectures
# =====================================================================
print("[Plot 3/3] Generating 4-Panel Confusion Matrix Grid...")
top_architectures = ["Linear SVM", "Random Forest", "Multi-Layer Perceptron (NN)", "Gradient Boosting"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
labels = sorted(y.unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cm_colormaps = ['Blues', 'Greens', 'Purples', 'Oranges']

for idx, model_name in enumerate(top_architectures):
    r, c = idx // 2, idx % 2
    ax = axes[r, c]
    
    pipe = Pipeline([('prep', preprocessor), ('clf', models[model_name])])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cm_colormaps[idx],
        xticklabels=[l.replace('_', ' ').capitalize() for l in labels],
        yticklabels=[l.replace('_', ' ').capitalize() for l in labels],
        ax=ax, cbar=False, linewidths=1.5, annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"{model_name}", weight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

fig.suptitle("Test Set Confusion Matrices Across Diverse Model Families", fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "Fig3_Confusion_Matrices_Diverse_Families.png")
plt.savefig(fig3_path)
plt.close()
print(f" -> Saved: {fig3_path}")

print("\n" + "="*70)
print(f"SUCCESS: All 13 models benchmarked!")
print(f"Leaderboard and publication plots saved in: {os.path.abspath(OUTPUT_DIR)}")
print("="*70)