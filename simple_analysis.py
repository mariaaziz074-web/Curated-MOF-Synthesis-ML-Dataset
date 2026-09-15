# simple_analysis.py
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('extracted_data/mof_cleaned_dataset.csv')

print("🔬 MOF Dataset Analysis")
print("=" * 50)
print(f"📊 Dataset shape: {df.shape}")
print(f"📄 Total papers processed: {len(df)}")
print(f"🔢 Total descriptors extracted: {df.shape[1]}")

print("\n📈 Data Availability:")
print("-" * 30)
for col in df.columns:
    if col not in ['filename', 'total_pages', 'total_characters']:
        non_null_count = df[col].count()
        if non_null_count > 0:
            percentage = (non_null_count / len(df)) * 100
            print(f"{col}: {non_null_count} papers ({percentage:.1f}%)")

# Temperature analysis
if 'temperature' in df.columns:
    temp_data = df['temperature'].dropna()
    if len(temp_data) > 0:
        print(f"\n🌡️ Temperature Statistics:")
        print(f"   Mean: {temp_data.mean():.1f}°C")
        print(f"   Range: {temp_data.min():.1f}°C - {temp_data.max():.1f}°C")
        print(f"   Most common: {temp_data.mode().iloc[0]:.1f}°C")

# Solvent analysis
if 'solvent' in df.columns:
    solvent_data = df['solvent'].dropna()
    print(f"\n🧪 Top 10 Solvents:")
    top_solvents = solvent_data.value_counts().head(10)
    for solvent, count in top_solvents.items():
        print(f"   {solvent}: {count} papers")

# Metal analysis
if 'metal' in df.columns:
    metal_data = df['metal'].dropna()
    print(f"\n⚗️ Metals Found:")
    metals = metal_data.value_counts()
    for metal, count in metals.items():
        print(f"   {metal}: {count} papers")

# pH analysis
if 'pH' in df.columns:
    pH_data = df['pH'].dropna()
    if len(pH_data) > 0:
        print(f"\n🧬 pH Statistics:")
        print(f"   Mean pH: {pH_data.mean():.2f}")
        print(f"   pH range: {pH_data.min():.1f} - {pH_data.max():.1f}")

# Section coverage
section_cols = [col for col in df.columns if col.startswith('has_')]
if section_cols:
    print(f"\n📖 Section Coverage:")
    for col in section_cols:
        section_name = col.replace('has_', '').replace('_section', '')
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"   {section_name.capitalize()}: {count} papers ({percentage:.1f}%)")

print(f"\n✅ Analysis complete! Files are in extracted_data/ folder")