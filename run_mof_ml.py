"""
MOF Machine Learning Pipeline: Unsupervised Clustering & Supervised Classification
Author: Curated MOF Project
Description: Robust ML experiments with strict CV, pipeline isolation, and publication plots.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# Configuration & Plot Styling (Journal-Grade)
# -------------------------------------------------------------
OUTPUT_DIR = "extracted_data/ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300
})

# -------------------------------------------------------------
# 1. Load and Standardize Dataset
# -------------------------------------------------------------
def load_and_preprocess(filepath="extracted_data/full_realistic_mof_data.csv"):
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows from {filepath}")
    
    # Chemistry normalization (canonical aliases)
    df['solvent'] = df['solvent'].astype(str).str.lower().str.strip()
    df['solvent'] = df['solvent'].replace({'h2o': 'water', 'nan': np.nan, 'none': np.nan})
    
    df['metal'] = df['metal'].astype(str).str.lower().str.strip()
    df['metal'] = df['metal'].replace({'nan': np.nan, 'none': np.nan})
    
    df['linker'] = df['linker'].astype(str).str.lower().str.strip()
    df['linker'] = df['linker'].replace({'h3btc': 'btc', 'terephthalic': 'bdc', 'nan': np.nan, 'none': np.nan})
    
    # Ensure numeric types
    for col in ['temperature', 'pH', 'time_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# -------------------------------------------------------------
# 2. Experiment 1: Unsupervised Dimensionality & Clustering
# -------------------------------------------------------------
def run_unsupervised_clustering(df):
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Chemical Space PCA & Clustering")
    print("="*60)
    
    feature_df = df[['metal', 'solvent', 'linker', 'temperature', 'pH', 'time_hours']].copy()
    
    # Pipeline: One-hot encode categoricals, impute and scale numerics
    cat_cols = ['metal', 'solvent', 'linker']
    num_cols = ['temperature', 'pH', 'time_hours']
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(feature_df)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_processed)
    var_exp = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1 = {var_exp[0]*100:.1f}%, PC2 = {var_exp[1]*100:.1f}% (Total: {sum(var_exp)*100:.1f}%)")
    
    # K-Means Clustering (k=3 chemical families)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_processed)
    df['Cluster'] = [f"Cluster {c+1}" for c in clusters]
    
    # Plot PCA & Clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by Cluster
    palette_clusters = sns.color_palette("tab10", 3)
    for c_id, c_name in enumerate(['Cluster 1', 'Cluster 2', 'Cluster 3']):
        idx = (clusters == c_id)
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], label=c_name, s=70, alpha=0.85, edgecolors='k', linewidth=0.5, color=palette_clusters[c_id])
    ax1.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}% Variance)")
    ax1.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}% Variance)")
    ax1.set_title("A. K-Means Synthesis Regimes (k=3)", weight='bold')
    ax1.legend(frameon=True, facecolor='white', loc='best')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: Colored by Top Metal
    top_metals = df['metal'].value_counts().nlargest(3).index.tolist()
    metal_labels = df['metal'].apply(lambda x: x if x in top_metals else 'other')
    unique_metals = list(dict.fromkeys(metal_labels))
    palette_metals = sns.color_palette("Set2", len(unique_metals))
    for m_id, m_name in enumerate(unique_metals):
        idx = (metal_labels == m_name)
        ax2.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Metal: {m_name.upper()}", s=70, alpha=0.85, edgecolors='k', linewidth=0.5, color=palette_metals[m_id])
    ax2.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}% Variance)")
    ax2.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}% Variance)")
    ax2.set_title("B. MOF Space by Metal Center", weight='bold')
    ax2.legend(frameon=True, facecolor='white', loc='best')
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Fig1_PCA_Clustering_Space.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] {plot_path}")
    
    return X_processed, preprocessor

# -------------------------------------------------------------
# 3. Experiment 2: Solvent Classification Benchmark
# -------------------------------------------------------------
def run_solvent_classification(df):
    print("\n" + "="*60)
    print("  EXPERIMENT 2: Solvent Prediction Benchmark")
    print("="*60)
    
    # Select samples with known solvent
    df_valid = df[df['solvent'].notna() & (df['solvent'] != 'nan')].copy()
    
    # Target grouping: Top 2 solvents (water, ethanol) vs Other
    top_solvents = ['water', 'ethanol']
    df_valid['target_solvent'] = df_valid['solvent'].apply(lambda x: x if x in top_solvents else 'other_solvent')
    
    print(f"Target distribution (Solvent):\n{df_valid['target_solvent'].value_counts()}")
    
    X = df_valid[['metal', 'linker', 'temperature', 'pH', 'time_hours']]
    y = df_valid['target_solvent']
    
    cat_cols = ['metal', 'linker']
    num_cols = ['temperature', 'pH', 'time_hours']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols)
    ])
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', C=1.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)
    }
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    benchmark_results = []
    
    for name, model in models.items():
        clf_pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', model)
        ])
        
        cv_res = cross_validate(
            clf_pipeline, X, y, cv=cv,
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=False
        )
        
        res = {
            "Model": name,
            "Accuracy_Mean": np.mean(cv_res['test_accuracy']),
            "Accuracy_Std": np.std(cv_res['test_accuracy']),
            "F1_Macro_Mean": np.mean(cv_res['test_f1_macro']),
            "F1_Macro_Std": np.std(cv_res['test_f1_macro']),
            "Precision_Mean": np.mean(cv_res['test_precision_macro']),
            "Recall_Mean": np.mean(cv_res['test_recall_macro'])
        }
        benchmark_results.append(res)
        print(f"[{name}] Acc: {res['Accuracy_Mean']:.3f} (+/- {res['Accuracy_Std']:.3f}) | F1: {res['F1_Macro_Mean']:.3f}")
        
    df_res = pd.DataFrame(benchmark_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "solvent_classification_benchmark.csv"), index=False)
    
    # Train-test split for Confusion Matrix of Best Model (Random Forest)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf_pipe = Pipeline([('prep', preprocessor), ('model', models['Random Forest'])])
    rf_pipe.fit(X_tr, y_tr)
    y_pred = rf_pipe.predict(X_te)
    
    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot A: Model Comparison
    x_indices = np.arange(len(df_res))
    ax1.bar(x_indices, df_res['Accuracy_Mean'], yerr=df_res['Accuracy_Std'], capsize=5, color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(df_res['Model'], rotation=15, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Cross-Validated Accuracy")
    ax1.set_title("A. Solvent Predictor Benchmark (4-Fold CV)", weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(df_res['Accuracy_Mean']):
        ax1.text(i, v + df_res['Accuracy_Std'][i] + 0.03, f"{v:.2f}", ha='center', fontweight='bold', fontsize=10)
        
    # Subplot B: Confusion Matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax2, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    ax2.set_xlabel("Predicted Solvent")
    ax2.set_ylabel("True Solvent")
    ax2.set_title("B. Test Set Confusion Matrix (Random Forest)", weight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Fig2_Solvent_Classification.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] {plot_path}")
    
    return df_res

# -------------------------------------------------------------
# 4. Experiment 3: Metal Center Feature Importance
# -------------------------------------------------------------
def run_metal_classification(df):
    print("\n" + "="*60)
    print("  EXPERIMENT 3: Metal Center Classification & Feature Importance")
    print("="*60)
    
    # Top metals: mg, ti vs others
    top_metals = ['mg', 'ti']
    df_metal = df.copy()
    df_metal['target_metal'] = df_metal['metal'].apply(lambda x: x if x in top_metals else 'other_metal')
    
    print(f"Target distribution (Metal):\n{df_metal['target_metal'].value_counts()}")
    
    X = df_metal[['solvent', 'linker', 'temperature', 'pH', 'time_hours']]
    y = df_metal['target_metal']
    
    cat_cols = ['solvent', 'linker']
    num_cols = ['temperature', 'pH', 'time_hours']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols)
    ])
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', rf)
    ])
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    print(f"[Metal RF Classifier] CV Accuracy: {np.mean(cv_res['test_accuracy']):.3f} (+/- {np.std(cv_res['test_accuracy']):.3f})")
    
    # Fit on all to inspect Feature Importances
    pipe.fit(X, y)
    feature_names = num_cols + list(pipe.named_steps['prep'].named_transformers_['cat'].named_steps['enc'].get_feature_names_out(cat_cols))
    importances = pipe.named_steps['model'].feature_importances_
    
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(8)
    feat_df.to_csv(os.path.join(OUTPUT_DIR, "metal_feature_importances.csv"), index=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=feat_df, x='Importance', y='Feature', palette='mako')
    plt.title("Key Predictors of Target Metal Node (Gini Importance)", weight='bold')
    plt.xlabel("Relative Importance")
    plt.ylabel("Synthesis Feature")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.005, p.get_y() + p.get_height() / 2, f"{width:.3f}", va='center', fontsize=10, weight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Fig3_Metal_Feature_Importance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] {plot_path}")

# -------------------------------------------------------------
# Main Execution Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("      MOF MACHINE LEARNING SUITE: TRAINING & VALIDATION")
    print("="*60)
    
    dataset = load_and_preprocess()
    
    # Run ML experiments
    run_unsupervised_clustering(dataset)
    run_solvent_classification(dataset)
    run_metal_classification(dataset)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] All ML experiments completed!")
    print(f"Results & figures saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60)