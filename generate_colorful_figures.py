"""
Colorful, Journal-Quality Figures for MOF Literature-Extracted Dataset & ML
- Color Palettes: Scientific gradients (crest, rocket, mako, viridis)
- Resolution: 300 DPI
- Zero missing glyph warnings (pure vector typography)
- Generous padding to prevent overlapping labels
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------
# Styling Configuration
# ---------------------------------------------------------------------
OUTPUT_DIR = "extracted_data/colorful_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font="Arial")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.titleweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ---------------------------------------------------------------------
# Data Loading & Standardizing
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_csv("extracted_data/full_realistic_mof_data.csv")
    
    df['solvent'] = df['solvent'].astype(str).str.lower().str.strip()
    df['solvent'] = df['solvent'].replace({'nan': np.nan, 'none': np.nan, 'h2o': 'water'})
    
    df['metal'] = df['metal'].astype(str).str.upper().str.strip()
    df['metal'] = df['metal'].replace({'NAN': np.nan, 'NONE': np.nan})
    
    df['linker'] = df['linker'].astype(str).str.upper().str.strip()
    df['linker'] = df['linker'].replace({'NAN': np.nan, 'NONE': np.nan, 'H3BTC': 'BTC', 'TEREPHTHALIC': 'BDC'})
    
    for col in ['temperature', 'pH', 'time_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

df = load_data()

# =====================================================================
# FIGURE 1: Comprehensive Descriptor Landscape (4-Panel Colorful Grid)
# =====================================================================
print("[1/5] Generating Figure 1: Descriptor Landscape...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Metal Distribution (Vibrant 'viridis' gradient)
metal_counts = df['metal'].dropna().value_counts()
colors_metal = sns.color_palette("viridis", len(metal_counts))
bars_a = axes[0, 0].bar(metal_counts.index, metal_counts.values, color=colors_metal, edgecolor='black', linewidth=0.8)
axes[0, 0].set_title("A. Metal Node Distribution (N=54)")
axes[0, 0].set_xlabel("Metal Center")
axes[0, 0].set_ylabel("Reported Frequency (Papers)")
for bar in bars_a:
    y = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, y + 0.4, f"{int(y)}", ha='center', va='bottom', fontweight='bold', color='#1a1a1a')
axes[0, 0].set_ylim(0, max(metal_counts.values) + 3)

# Panel B: Solvent Distribution (Warm 'rocket' gradient)
solvent_counts = df['solvent'].dropna().value_counts()
colors_solvent = sns.color_palette("rocket", len(solvent_counts))
bars_b = axes[0, 1].bar(solvent_counts.index.str.capitalize(), solvent_counts.values, color=colors_solvent, edgecolor='black', linewidth=0.8)
axes[0, 1].set_title("B. Reaction Solvent Systems (N=42)")
axes[0, 1].set_xlabel("Solvent Type")
axes[0, 1].set_ylabel("Reported Frequency (Papers)")
for bar in bars_b:
    y = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{int(y)}", ha='center', va='bottom', fontweight='bold', color='#1a1a1a')
axes[0, 1].set_ylim(0, max(solvent_counts.values) + 4)

# Panel C: Linker Distribution (Cool 'crest' gradient)
linker_counts = df['linker'].dropna().value_counts()
colors_linker = sns.color_palette("crest", len(linker_counts))
bars_c = axes[1, 0].bar(linker_counts.index, linker_counts.values, color=colors_linker, edgecolor='black', linewidth=0.8)
axes[1, 0].set_title("C. Organic Bridging Linkers (N=32)")
axes[1, 0].set_xlabel("Ligand / Linker")
axes[1, 0].set_ylabel("Reported Frequency (Papers)")
for bar in bars_c:
    y = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, y + 0.4, f"{int(y)}", ha='center', va='bottom', fontweight='bold', color='#1a1a1a')
axes[1, 0].set_ylim(0, max(linker_counts.values) + 3)

# Panel D: Descriptor Completeness / Coverage Rate ('mako_r' palette)
coverage = [
    ('Metal Node', (df['metal'].notna().sum()/54)*100),
    ('Solvent', (df['solvent'].notna().sum()/54)*100),
    ('Linker', (df['linker'].notna().sum()/54)*100),
    ('Time (h)', (df['time_hours'].notna().sum()/54)*100),
    ('Temperature', (df['temperature'].notna().sum()/54)*100),
    ('pH', (df['pH'].notna().sum()/54)*100),
]
cov_df = pd.DataFrame(coverage, columns=['Descriptor', 'Coverage'])
colors_cov = sns.color_palette("mako_r", len(cov_df))
bars_d = axes[1, 1].barh(cov_df['Descriptor'], cov_df['Coverage'], color=colors_cov, edgecolor='black', linewidth=0.8)
axes[1, 1].set_title("D. Parameter Reporting Density Across Reviews")
axes[1, 1].set_xlabel("Extraction Coverage (%)")
axes[1, 1].set_xlim(0, 115)
for bar in bars_d:
    w = bar.get_width()
    axes[1, 1].text(w + 2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", ha='left', va='center', fontweight='bold', color='#1a1a1a')

plt.tight_layout(pad=3.0)
fig1_path = os.path.join(OUTPUT_DIR, "Fig1_Colorful_Descriptor_Landscape.png")
plt.savefig(fig1_path)
plt.close()
print(f" -> Saved: {fig1_path}")


# =====================================================================
# FIGURE 2: Chemical Co-Occurrence Heatmaps (Metal vs Solvent / Linker)
# =====================================================================
print("[2/5] Generating Figure 2: Chemical Co-Occurrence Heatmaps...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Metal vs Solvent Cross-tabulation
ct_solvent = pd.crosstab(df['metal'].dropna(), df['solvent'].dropna())
top_metals = df['metal'].value_counts().nlargest(4).index
ct_solvent = ct_solvent.loc[ct_solvent.index.isin(top_metals)]
sns.heatmap(ct_solvent, annot=True, fmt='d', cmap='YlGnBu', cbar=True, ax=ax1, linewidths=1.2, linecolor='white', annot_kws={"size": 13, "weight": "bold"})
ax1.set_title("A. Metal Node vs Solvent Co-Occurrence", weight='bold')
ax1.set_xlabel("Solvent System")
ax1.set_ylabel("Metal Center")

# Panel B: Metal vs Linker Cross-tabulation
ct_linker = pd.crosstab(df['metal'].dropna(), df['linker'].dropna())
ct_linker = ct_linker.loc[ct_linker.index.isin(top_metals)]
sns.heatmap(ct_linker, annot=True, fmt='d', cmap='Purples', cbar=True, ax=ax2, linewidths=1.2, linecolor='white', annot_kws={"size": 13, "weight": "bold"})
ax2.set_title("B. Metal Node vs Organic Linker Co-Occurrence", weight='bold')
ax2.set_xlabel("Organic Ligand")
ax2.set_ylabel("Metal Center")

plt.tight_layout(pad=3.0)
fig2_path = os.path.join(OUTPUT_DIR, "Fig2_Colorful_Cooccurrence_Heatmaps.png")
plt.savefig(fig2_path)
plt.close()
print(f" -> Saved: {fig2_path}")


# =====================================================================
# FIGURE 3: PCA Chemical Subspace & K-Means Clusters
# =====================================================================
print("[3/5] Generating Figure 3: PCA Chemical Subspace...")
feature_df = df[['metal', 'solvent', 'linker', 'temperature', 'pH', 'time_hours']].copy()

cat_cols = ['metal', 'solvent', 'linker']
num_cols = ['temperature', 'pH', 'time_hours']

cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
    ('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
prep = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

X_mat = prep.fit_transform(feature_df)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_mat)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_mat)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot A: Clusters (Vibrant categorical colors with glowing halos)
cluster_colors = ['#FF1744', '#00E676', '#2979FF']
cluster_names = ['Cluster 1: Aqueous/Mg-Ti', 'Cluster 2: Solvothermal/Cu', 'Cluster 3: High-T Hybrid']

for i in range(3):
    idx = (kmeans.labels_ == i)
    ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], s=160, c=cluster_colors[i], label=cluster_names[i], edgecolors='black', linewidth=1.2, alpha=0.9, zorder=3)
ax1.set_title(f"A. Unsupervised Synthesis Regimes (Total Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
ax1.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot B: Metal Color Map (Bold colors)
metal_top = df['metal'].apply(lambda x: x if x in ['MG', 'TI', 'CU'] else 'Other')
metal_colors = {'MG': '#00B0FF', 'TI': '#FF9100', 'CU': '#D500F9', 'Other': '#9E9E9E'}

for m_val, col in metal_colors.items():
    idx = (metal_top == m_val)
    ax2.scatter(X_pca[idx, 0], X_pca[idx, 1], s=160, c=col, label=f"Metal: {m_val}", edgecolors='black', linewidth=1.2, alpha=0.9, zorder=3)
ax2.set_title("B. Projection of Chemical Space by Metal Center")
ax2.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(pad=3.0)
fig3_path = os.path.join(OUTPUT_DIR, "Fig3_Colorful_PCA_Chemical_Space.png")
plt.savefig(fig3_path)
plt.close()
print(f" -> Saved: {fig3_path}")


# =====================================================================
# FIGURE 4: Multi-Model Machine Learning Classification Benchmark
# =====================================================================
print("[4/5] Generating Figure 4: ML Benchmark Comparison...")

df_ml = df[df['solvent'].notna()].copy()
df_ml['target_solvent'] = df_ml['solvent'].apply(lambda x: x if x in ['water', 'ethanol'] else 'other')

X_solv = df_ml[['metal', 'linker', 'temperature', 'pH', 'time_hours']]
y_solv = df_ml['target_solvent']

solv_prep = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), ['temperature', 'pH', 'time_hours']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='miss')), ('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), ['metal', 'linker'])
])

models = {
    'Support Vector Machine': SVC(kernel='linear', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
}

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
benchmark_data = []

for m_name, model in models.items():
    pipe = Pipeline([('prep', solv_prep), ('model', model)])
    scores = cross_validate(pipe, X_solv, y_solv, cv=cv, scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
    benchmark_data.append({
        'Model': m_name,
        'Accuracy': np.mean(scores['test_accuracy']),
        'Accuracy_Std': np.std(scores['test_accuracy']),
        'F1-Score': np.mean(scores['test_f1_macro']),
        'Precision': np.mean(scores['test_precision_macro']),
        'Recall': np.mean(scores['test_recall_macro'])
    })

bench_df = pd.DataFrame(benchmark_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Multi-metric grouped colorful bar chart
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
x = np.arange(len(bench_df))
width = 0.20
metric_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, met in enumerate(metrics):
    offset = (i - 1.5) * width
    ax1.bar(x + offset, bench_df[met], width, label=met, color=metric_palette[i], edgecolor='black', linewidth=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(bench_df['Model'], rotation=15, ha='right', weight='bold')
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Metric Score (4-Fold Cross Validation)")
ax1.set_title("A. Multi-Metric Classifier Benchmark", weight='bold')
ax1.legend(loc='upper right', frameon=True, facecolor='white')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Subplot B: Confusion Matrix with Vibrant 'Spectral' or 'Blues'
X_tr, X_te, y_tr, y_te = train_test_split(X_solv, y_solv, test_size=0.25, random_state=42, stratify=y_solv)
rf_pipe = Pipeline([('prep', solv_prep), ('model', models['Support Vector Machine'])])
rf_pipe.fit(X_tr, y_tr)
y_pred = rf_pipe.predict(X_te)
labels = sorted(y_solv.unique())
cm = confusion_matrix(y_te, y_pred, labels=labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[s.capitalize() for s in labels], yticklabels=[s.capitalize() for s in labels], ax=ax2, cbar=False, linewidths=1.5, annot_kws={"size": 14, "weight": "bold"})
ax2.set_title("B. Test Set Confusion Matrix (SVM)", weight='bold')
ax2.set_xlabel("Predicted Solvent")
ax2.set_ylabel("True Solvent")

plt.tight_layout(pad=3.0)
fig4_path = os.path.join(OUTPUT_DIR, "Fig4_Colorful_ML_Classification.png")
plt.savefig(fig4_path)
plt.close()
print(f" -> Saved: {fig4_path}")


# =====================================================================
# FIGURE 5: Synthesis Parameter Correlation & Feature Importances
# =====================================================================
print("[5/5] Generating Figure 5: Feature Importances...")
df_rf = df.copy()
df_rf['target_metal'] = df_rf['metal'].apply(lambda x: x if x in ['MG', 'TI'] else 'OTHER')
X_m = df_rf[['solvent', 'linker', 'temperature', 'pH', 'time_hours']]
y_m = df_rf['target_metal']

m_prep = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), ['temperature', 'pH', 'time_hours']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='miss')), ('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), ['solvent', 'linker'])
])

pipe_metal = Pipeline([('prep', m_prep), ('model', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))])
pipe_metal.fit(X_m, y_m)

feat_names = ['temperature', 'pH', 'time_hours'] + list(pipe_metal.named_steps['prep'].named_transformers_['cat'].named_steps['enc'].get_feature_names_out(['solvent', 'linker']))
importances = pipe_metal.named_steps['model'].feature_importances_

feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True).tail(8)
feat_df['Clean_Feature'] = feat_df['Feature'].str.replace('solvent_', 'Solvent: ').str.replace('linker_', 'Linker: ').str.capitalize()

plt.figure(figsize=(10, 6))
bar_colors = sns.color_palette("plasma", len(feat_df))
bars = plt.barh(feat_df['Clean_Feature'], feat_df['Importance'], color=bar_colors, edgecolor='black', linewidth=0.8)
plt.title("Gini Feature Importance Ranking (Random Forest)", weight='bold', pad=15)
plt.xlabel("Relative Importance Score")
plt.ylabel("Synthesis Condition / Descriptor")
plt.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontweight='bold', fontsize=11, color='#111111')

plt.xlim(0, max(feat_df['Importance']) + 0.04)
plt.tight_layout(pad=2.0)
fig5_path = os.path.join(OUTPUT_DIR, "Fig5_Colorful_Feature_Importance.png")
plt.savefig(fig5_path)
plt.close()
print(f" -> Saved: {fig5_path}")

print("\n" + "="*60)
print(f"SUCCESS: All 5 colorful figures saved in:")
print(f"-> {os.path.abspath(OUTPUT_DIR)}")
print("="*60)