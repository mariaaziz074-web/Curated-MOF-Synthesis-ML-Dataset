# quick_analysis.py
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('extracted_data/mof_cleaned_dataset.csv')

print("Dataset shape:", df.shape)
print("\nColumns with data:")
for col in df.columns:
    non_null_count = df[col].count()
    if non_null_count > 0:
        print(f"{col}: {non_null_count} papers")

# Plot temperature distribution
if 'temperature' in df.columns:
    df['temperature'].dropna().hist(bins=20)
    plt.title('Temperature Distribution in MOF Synthesis')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.show()

# Show top solvents
if 'solvent' in df.columns:
    print("\nTop 10 solvents:")
    print(df['solvent'].value_counts().head(10))