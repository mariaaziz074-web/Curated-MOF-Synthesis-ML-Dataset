# ml_ready_dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

def create_ml_ready_dataset():
    """Create a complete ML-ready dataset"""
    
    # Load the realistic data
    df = pd.read_csv('extracted_data/full_realistic_mof_data.csv')
    print(f"📊 Starting with {df.shape[0]} papers and {df.shape[1]} features")
    
    # Step 1: Handle missing values intelligently
    print("\n🔧 Step 1: Handling Missing Values")
    
    # Separate numeric and categorical columns
    numeric_cols = ['temperature', 'pH', 'synthesis_temp', 'time_hours', 'time_minutes', 
                   'yield', 'surface_area', 'pressure', 'time_total_hours']
    categorical_cols = ['solvent', 'metal', 'linker']
    
    # Only keep columns that exist in our data
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    print(f"  Numeric columns: {numeric_cols}")
    print(f"  Categorical columns: {categorical_cols}")
    
    # Step 2: Create features from categorical data
    print("\n🔧 Step 2: Encoding Categorical Variables")
    
    # One-hot encode categorical variables
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            # Drop original column
            df_encoded.drop(col, axis=1, inplace=True)
    
    # Step 3: Handle numeric missing values
    print("\n🔧 Step 3: Imputing Missing Values")
    
    for col in numeric_cols:
        if col in df_encoded.columns:
            # Use median imputation for numeric data
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
    
    # Step 4: Create derived features
    print("\n🔧 Step 4: Feature Engineering")
    
    # Temperature-based features
    if 'temperature' in df_encoded.columns:
        df_encoded['temp_category'] = pd.cut(df_encoded['temperature'], 
                                           bins=[0, 100, 200, 300, 1000], 
                                           labels=['low', 'medium', 'high', 'very_high'])
        df_encoded = pd.concat([df_encoded, pd.get_dummies(df_encoded['temp_category'], prefix='temp')], axis=1)
        df_encoded.drop('temp_category', axis=1, inplace=True)
    
    # pH-based features
    if 'pH' in df_encoded.columns:
        df_encoded['pH_acidic'] = (df_encoded['pH'] < 7).astype(int)
        df_encoded['pH_basic'] = (df_encoded['pH'] > 7).astype(int)
        df_encoded['pH_neutral'] = ((df_encoded['pH'] >= 6.5) & (df_encoded['pH'] <= 7.5)).astype(int)
    
    # Step 5: Remove filename and create feature matrix
    feature_cols = [col for col in df_encoded.columns if col != 'filename']
    X = df_encoded[feature_cols]
    
    # Step 6: Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols)
    
    # Step 7: Scale features
    print("\n🔧 Step 5: Feature Scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Step 8: Create target variables for different ML tasks
    print("\n🔧 Step 6: Creating Target Variables")
    
    # Example targets (you can modify based on your research goals)
    targets = {}
    
    # Target 1: High/Low temperature synthesis
    if 'temperature' in df.columns:
        temp_median = df['temperature'].median()
        targets['high_temp_synthesis'] = (df['temperature'] > temp_median).astype(int)
    
    # Target 2: Successful synthesis (has multiple descriptors)
    descriptor_count = df[feature_cols].notna().sum(axis=1)
    targets['rich_data_paper'] = (descriptor_count > descriptor_count.median()).astype(int)
    
    # Target 3: Specific metal prediction
    if 'metal' in df.columns:
        targets['metal_category'] = df['metal'].fillna('unknown')
    
    # Step 9: Save ML-ready datasets
    print("\n💾 Step 7: Saving ML-Ready Files")
    
    # Save feature matrix
    X_scaled.to_csv('extracted_data/ml_features_scaled.csv', index=False)
    X.to_csv('extracted_data/ml_features_raw.csv', index=False)
    
    # Save targets
    targets_df = pd.DataFrame(targets)
    targets_df.to_csv('extracted_data/ml_targets.csv', index=False)
    
    # Save original data with filenames for reference
    df_encoded['filename'] = df['filename']
    df_encoded.to_csv('extracted_data/ml_complete_dataset.csv', index=False)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'extracted_data/feature_scaler.pkl')
    joblib.dump(imputer, 'extracted_data/feature_imputer.pkl')
    
    # Step 10: Create train/test splits
    print("\n🔧 Step 8: Creating Train/Test Splits")
    
    for target_name, target_values in targets.items():
        if len(target_values.dropna()) > 10:  # Only if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, target_values, test_size=0.2, random_state=42, stratify=target_values)
            
            # Save splits
            X_train.to_csv(f'extracted_data/X_train_{target_name}.csv', index=False)
            X_test.to_csv(f'extracted_data/X_test_{target_name}.csv', index=False)
            pd.Series(y_train).to_csv(f'extracted_data/y_train_{target_name}.csv', index=False)
            pd.Series(y_test).to_csv(f'extracted_data/y_test_{target_name}.csv', index=False)
    
    # Summary report
    print(f"\n🎉 ML-Ready Dataset Created Successfully!")
    print(f"=" * 50)
    print(f"📊 Dataset Summary:")
    print(f"  • Original papers: {df.shape[0]}")
    print(f"  • Final features: {X_scaled.shape[1]}")
    print(f"  • Feature types: {len(numeric_cols)} numeric + {len([c for c in X.columns if any(cat in c for cat in categorical_cols)])} categorical")
    print(f"  • Missing values: {X.isnull().sum().sum()} (should be 0)")
    print(f"  • Target variables: {len(targets)}")
    print(f"  • Ready for: Classification, Regression, Clustering")
    
    return X_scaled, targets_df, df_encoded

if __name__ == "__main__":
    X, y, full_data = create_ml_ready_dataset()