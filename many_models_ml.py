"""
Comprehensive ML Comparison for MOF Dataset
Tests 9 different Machine Learning Algorithms
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

# All Classical ML Models
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create output folder
OUTPUT_DIR = "extracted_data/ml_many_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set publication style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load Data
df = pd.read_csv("extracted_data/full_realistic_mof_data.csv")

# Clean data
df['solvent'] = df['solvent'].astype(str).str.lower().str.strip()
df['solvent'] = df['solvent'].replace({'nan': np.nan, 'none': np.nan, 'h2o': 'water'})

df['metal'] = df['metal'].astype(str).str.lower().str.strip()
df['metal'] = df['metal'].replace({'nan': np.nan})

df['linker'] = df['linker'].astype(str).str.lower().str.strip()
df['linker'] = df['linker'].replace({'nan': np.nan, 'h3btc': 'btc', 'terephthalic': 'bdc'})

for col in ['temperature', 'pH', 'time_hours']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("=" * 60)
print("COMPREHENSIVE ML MODEL COMPARISON")
print("=" * 60)

# Prepare Solvent Prediction
df_solv = df[df['solvent'].notna()].copy()
top_solvents = ['water', 'ethanol']
df_solv['target'] = df_solv['solvent'].apply(lambda x: x if x in top_solvents else 'other')

X = df_solv[['metal', 'linker', 'temperature', 'pH', 'time_hours']]
y = df_solv['target']

print(f"Samples with known solvent: {len(X)}")
print(f"Target classes: {y.value_counts().to_dict()}")
print("=" * 60)

# Preprocessor
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

# Define 9 Different ML Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Ridge Classifier": RidgeClassifier(random_state=42)
}

print("Testing 12 Machine Learning Models...")
print("-" * 60)

results = []
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

for name, model in models.items():
    try:
        pipe = Pipeline([('prep', preprocessor), ('clf', model)])
        
        cv_scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        )
        
        acc_mean = np.mean(cv_scores['test_accuracy'])
        acc_std = np.std(cv_scores['test_accuracy'])
        f1_mean = np.mean(cv_scores['test_f1_macro'])
        
        results.append({
            'Model': name,
            'Accuracy': acc_mean,
            'Acc_Std': acc_std,
            'F1_Macro': f1_mean,
            'Precision': np.mean(cv_scores['test_precision_macro']),
            'Recall': np.mean(cv_scores['test_recall_macro'])
        })
        
        print(f"{name:28s} | Acc = {acc_mean:.3f} ± {acc_std:.3f} | F1 = {f1_mean:.3f}")
    
    except Exception as e:
        print(f"{name:28s} | Skipped ({str(e)[:30]})")

print("-" * 60)

# Create Results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "all_models_comparison.csv"), index=False)

print(f"\nBest Performing Model: {results_df.iloc[0]['Model']}")
print(f"Best Accuracy: {results_df.iloc[0]['Accuracy']:.3f} ± {results_df.iloc[0]['Acc_Std']:.3f}")
print(f"Best F1-Score: {results_df.iloc[0]['F1_Macro']:.3f}")

# Colorful Bar Plot
plt.figure(figsize=(12, 6))
colors = sns.color_palette("viridis", len(results_df))
bars = plt.barh(results_df['Model'], results_df['Accuracy'], xerr=results_df['Acc_Std'],
                color=colors, edgecolor='black', linewidth=0.8, capsize=5, alpha=0.95)

plt.title("Solvent Prediction: Comparison of 12 ML Algorithms (4-Fold CV)")
plt.xlabel("Cross-Validation Accuracy")
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.4)

for bar, acc, std in zip(bars, results_df['Accuracy'], results_df['Acc_Std']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{acc:.3f} ± {std:.3f}", va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_All_Models_Comparison.png"))
plt.close()

# Confusion Matrix for Best Model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=42, stratify=y)

best_pipe = Pipeline([('prep', preprocessor), ('clf', best_model)])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[l.capitalize() for l in labels],
            yticklabels=[l.capitalize() for l in labels],
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=1.5)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Solvent")
plt.ylabel("True Solvent")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Best_Model_Confusion_Matrix.png"))
plt.close()

print("\n" + "=" * 60)
print(f"All Results Saved in: extracted_data/ml_many_models/")
print("  - Fig1_All_Models_Comparison.png")
print("  - Fig2_Best_Model_Confusion_Matrix.png")
print("  - all_models_comparison.csv")
print("=" * 60)