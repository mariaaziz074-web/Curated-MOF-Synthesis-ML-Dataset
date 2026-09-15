# data_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset
df = pd.read_csv('extracted_data/mof_cleaned_dataset.csv')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('MOF Dataset Analysis - 54 Research Papers', fontsize=16, fontweight='bold')

# 1. Temperature Distribution
if 'temperature' in df.columns:
    temp_data = df['temperature'].dropna()
    if len(temp_data) > 0:
        axes[0,0].hist(temp_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title(f'Temperature Distribution\n({len(temp_data)} papers)')
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Number of Papers')
        axes[0,0].axvline(temp_data.mean(), color='red', linestyle='--', label=f'Mean: {temp_data.mean():.1f}°C')
        axes[0,0].legend()

# 2. Top Solvents
if 'solvent' in df.columns:
    solvent_data = df['solvent'].dropna()
    top_solvents = solvent_data.value_counts().head(8)
    if len(top_solvents) > 0:
        top_solvents.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title(f'Top 8 Solvents\n({len(solvent_data)} papers with solvent data)')
        axes[0,1].set_xlabel('Solvents')
        axes[0,1].set_ylabel('Number of Papers')
        axes[0,1].tick_params(axis='x', rotation=45)

# 3. pH Distribution
if 'pH' in df.columns:
    pH_data = df['pH'].dropna()
    if len(pH_data) > 0:
        axes[0,2].hist(pH_data, bins=10, alpha=0.7, color='coral', edgecolor='black')
        axes[0,2].set_title(f'pH Distribution\n({len(pH_data)} papers)')
        axes[0,2].set_xlabel('pH Value')
        axes[0,2].set_ylabel('Number of Papers')
        axes[0,2].axvline(pH_data.mean(), color='blue', linestyle='--', label=f'Mean: {pH_data.mean():.2f}')
        axes[0,2].legend()

# 4. Metals Distribution
if 'metal' in df.columns:
    metal_data = df['metal'].dropna()
    if len(metal_data) > 0:
        metal_counts = metal_data.value_counts()
        axes[1,0].pie(metal_counts.values, labels=metal_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title(f'Metal Distribution\n({len(metal_data)} papers)')

# 5. Section Coverage
section_cols = [col for col in df.columns if col.startswith('has_')]
if section_cols:
    section_data = []
    section_names = []
    for col in section_cols:
        section_name = col.replace('has_', '').replace('_section', '').capitalize()
        count = df[col].sum()
        section_data.append(count)
        section_names.append(section_name)
    
    axes[1,1].bar(section_names, section_data, color='lightblue')
    axes[1,1].set_title('Section Coverage Across Papers')
    axes[1,1].set_xlabel('Paper Sections')
    axes[1,1].set_ylabel('Number of Papers')
    axes[1,1].tick_params(axis='x', rotation=45)

# 6. Data Completeness
numeric_cols = ['temperature', 'time', 'pH', 'pressure']
completeness_data = []
col_names = []
for col in numeric_cols:
    if col in df.columns:
        completeness = (df[col].count() / len(df)) * 100
        completeness_data.append(completeness)
        col_names.append(col.capitalize())

if completeness_data:
    axes[1,2].bar(col_names, completeness_data, color='gold')
    axes[1,2].set_title('Data Completeness')
    axes[1,2].set_xlabel('Descriptors')
    axes[1,2].set_ylabel('Completeness (%)')
    axes[1,2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('extracted_data/mof_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Visualizations created and saved as 'mof_analysis_plots.png'")