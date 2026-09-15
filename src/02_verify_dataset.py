# verify_dataset.py
import os
import pandas as pd
import numpy as np

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def verify_csv_dataset():
    csv_path = os.path.join("extracted_data", "full_realistic_mof_data.csv")
    
    if not os.path.exists(csv_path):
        # Fallback check
        csv_path = os.path.join("extracted_data", "mof_cleaned_dataset.csv")
        if not os.path.exists(csv_path):
            print("❌ Error: No CSV dataset found in 'extracted_data/' folder.")
            return

    print_header(f"🔍 INSPECTING DATASET: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Basic Shape and Memory
    print(f"• Total Rows (Papers):      {df.shape[0]}")
    print(f"• Total Columns (Features): {df.shape[1]}")
    print(f"• Memory Usage:             {df.memory_usage().sum() / 1024:.2f} KB")

    # 2. Data Types and Null Summary
    print_header("📋 COLUMN BREAKDOWN & MISSING VALUES")
    null_summary = pd.DataFrame({
        'Data Type': df.dtypes.astype(str),
        'Filled Count': df.notnull().sum(),
        'Missing Count': df.isnull().sum(),
        'Coverage (%)': (df.notnull().sum() / len(df) * 100).round(1)
    })
    print(null_summary.to_string())

    # 3. Chemical & Numerical Sanity Checks
    print_header("🧪 DOMAIN & NUMERICAL SANITY CHECKS (MOF Parameters)")
    
    warnings = []
    
    # Check Temperature
    temp_cols = [c for c in df.columns if 'temp' in c.lower()]
    for col in temp_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"🌡️  Column '{col}':")
                print(f"    Min: {vals.min()}°C | Median: {vals.median()}°C | Mean: {vals.mean():.1f}°C | Max: {vals.max()}°C")
                if vals.min() < 20:
                    warnings.append(f"Suspiciously low temperature in '{col}' (< 20°C)")
                if vals.max() > 500:
                    warnings.append(f"Suspiciously high temperature in '{col}' (> 500°C)")
    
    # Check pH
    if 'pH' in df.columns and pd.api.types.is_numeric_dtype(df['pH']):
        vals = df['pH'].dropna()
        if len(vals) > 0:
            print(f"🧪  Column 'pH':")
            print(f"    Min: {vals.min()} | Median: {vals.median()} | Mean: {vals.mean():.2f} | Max: {vals.max()}")
            if vals.min() < 0 or vals.max() > 14:
                warnings.append("pH values detected outside valid range [0, 14]")

    # Check Time
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    for col in time_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"⏱️  Column '{col}':")
                print(f"    Min: {vals.min()} h | Median: {vals.median()} h | Max: {vals.max()} h")

    # 4. Categorical Health & Value Distribution
    print_header("🔤 CATEGORICAL CARDINALITY & TOP VALUES")
    cat_cols = ['solvent', 'metal', 'linker']
    for col in cat_cols:
        if col in df.columns:
            counts = df[col].dropna().value_counts()
            print(f"\n• {col.upper()} (Unique values: {df[col].nunique()}):")
            for item, count in counts.head(5).items():
                print(f"    - {str(item).ljust(15)} : {count} papers ({(count/len(df)*100):.1f}%)")

    # 5. Duplicates Check
    print_header("🔍 DUPLICATE CHECKS")
    if 'filename' in df.columns:
        duplicate_files = df['filename'].duplicated().sum()
        print(f"• Duplicate filenames: {duplicate_files}")
        if duplicate_files > 0:
            warnings.append(f"{duplicate_files} duplicate paper filenames found.")

    feature_cols = [c for c in df.columns if c != 'filename']
    exact_duplicates = df.duplicated(subset=feature_cols).sum()
    print(f"• Identical feature rows: {exact_duplicates}")

    # 6. First 5 Rows Preview
    print_header("👀 DATA PREVIEW (First 5 Rows)")
    preview_cols = [c for c in ['filename', 'metal', 'solvent', 'linker', 'temperature', 'pH'] if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))

    # 7. Final ML Readiness Checklist
    print_header("🎯 ML READINESS SCORECARD")
    
    score_items = {
        "Dataset loaded successfully": True,
        "No infinite values": not np.isinf(df.select_dtypes(include=np.number)).any().any(),
        "Chemistry values within realistic limits": len(warnings) == 0,
        "Sufficient sample size for Classical ML (RF/XGBoost/SVM)": len(df) >= 30,
        "Target or feature variance exists": all(df[c].nunique() > 1 for c in df.columns if c != 'filename')
    }

    all_pass = True
    for item, passed in score_items.items():
        status = "✅ PASS" if passed else "⚠️ WARNING"
        if not passed:
            all_pass = False
        print(f"  [{status}] {item}")

    if warnings:
        print("\n⚠️ Warnings Found:")
        for w in warnings:
            print(f"   • {w}")

    print("\n" + "=" * 70)
    if all_pass:
        print("🚀 RESULT: CSV is clean and ready for feature encoding & ML modeling!")
    else:
        print("💡 RESULT: Ready for ML, but keep the warnings above in mind during preprocessing.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    verify_csv_dataset()