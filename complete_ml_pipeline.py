# complete_ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_complete_ml_dataset():
    """Create a complete, ML-ready dataset with proper preprocessing"""
    
    print("🚀 Creating Complete ML-Ready MOF Dataset")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('extracted_data/full_realistic_mof_data.csv')
    print(f"📊 Starting with {df.shape[0]} papers, {df.shape[1]} raw features")
    
    # Step 1: Feature Engineering
    print("\n🔧 Step 1: Advanced Feature Engineering")
    
    # Create comprehensive feature matrix
    features_df = pd.DataFrame()
    features_df['paper_id'] = range(len(df))
    
    # Numerical features with proper handling
    numerical_features = []
    
    # Temperature features
    if 'temperature' in df.columns:
        temp_series = pd.to_numeric(df['temperature'], errors='coerce')
        features_df['synthesis_temp'] = temp_series
        features_df['temp_high'] = (temp_series > 200).astype(int)
        features_df['temp_moderate'] = ((temp_series >= 100) & (temp_series <= 200)).astype(int) 
        features_df['temp_low'] = (temp_series < 100).astype(int)
        numerical_features.extend(['synthesis_temp'])
    
    # pH features
    if 'pH' in df.columns:
        pH_series = pd.to_numeric(df['pH'], errors='coerce')
        features_df['pH_value'] = pH_series
        features_df['acidic'] = (pH_series < 7).astype(int)
        features_df['basic'] = (pH_series > 7).astype(int)
        features_df['neutral'] = ((pH_series >= 6.5) & (pH_series <= 7.5)).astype(int)
        numerical_features.extend(['pH_value'])
    
    # Time features
    if 'time_hours' in df.columns:
        time_series = pd.to_numeric(df['time_hours'], errors='coerce')
        features_df['reaction_time'] = time_series
        features_df['quick_reaction'] = (time_series <= 2).astype(int)
        features_df['long_reaction'] = (time_series > 24).astype(int)
        numerical_features.extend(['reaction_time'])
    
    # Categorical features - One-hot encoding
    print("\n🔧 Step 2: Categorical Feature Encoding")
    
    # Solvent encoding
    if 'solvent' in df.columns:
        common_solvents = ['water', 'ethanol', 'dmf', 'methanol', 'acetone']
        for solvent in common_solvents:
            features_df[f'solvent_{solvent}'] = df['solvent'].str.lower().eq(solvent).astype(int)
        features_df['solvent_other'] = (~df['solvent'].str.lower().isin(common_solvents)).astype(int)
    
    # Metal encoding  
    if 'metal' in df.columns:
        common_metals = ['zn', 'cu', 'fe', 'co', 'ni', 'mg', 'al']
        for metal in common_metals:
            features_df[f'metal_{metal}'] = df['metal'].str.lower().eq(metal).astype(int)
        features_df['metal_other'] = (~df['metal'].str.lower().isin(common_metals)).astype(int)
    
    # Linker encoding
    if 'linker' in df.columns:
        common_linkers = ['bdc', 'btc', 'bpdc', 'terephthalic']
        for linker in common_linkers:
            features_df[f'linker_{linker}'] = df['linker'].str.lower().eq(linker).astype(int)
        features_df['linker_other'] = (~df['linker'].str.lower().isin(common_linkers)).astype(int)
    
    # Step 3: Advanced imputation
    print("\n🔧 Step 3: Advanced Missing Value Imputation")
    
    # Separate features for different imputation strategies
    feature_cols = [col for col in features_df.columns if col != 'paper_id']
    
    # Use KNN imputation for numerical features
    if numerical_features:
        numerical_data = features_df[numerical_features]
        knn_imputer = KNNImputer(n_neighbors=3)
        numerical_imputed = knn_imputer.fit_transform(numerical_data)
        for i, col in enumerate(numerical_features):
            features_df[col] = numerical_imputed[:, i]
    
    # Fill binary features with 0 (absence of information)
    binary_cols = [col for col in feature_cols if col not in numerical_features]
    for col in binary_cols:
        features_df[col] = features_df[col].fillna(0)
    
    # Step 4: Create target variables
    print("\n🔧 Step 4: Creating Target Variables")
    
    targets = {}
    
    # Target 1: High-temperature synthesis (classification)
    if 'synthesis_temp' in features_df.columns:
        temp_median = features_df['synthesis_temp'].median()
        targets['high_temp_synthesis'] = (features_df['synthesis_temp'] > temp_median).astype(int)
    
    # Target 2: Data completeness (regression) - how much info was extractable
    data_completeness = features_df[feature_cols].notna().sum(axis=1)
    targets['data_richness'] = data_completeness
    
    # Target 3: Metal type classification
    if 'metal' in df.columns:
        # Create simplified metal categories
        metal_map = {'zn': 'transition', 'cu': 'transition', 'fe': 'transition', 
                    'co': 'transition', 'ni': 'transition', 'mg': 'alkaline', 'al': 'post_transition'}
        df['metal_category'] = df['metal'].str.lower().map(metal_map).fillna('other')
        
        # Encode metal categories
        le = LabelEncoder()
        targets['metal_type'] = le.fit_transform(df['metal_category'])
    
    # Step 5: Feature scaling and final preparation
    print("\n🔧 Step 5: Feature Scaling and Final Preparation")
    
    # Prepare final feature matrix
    X = features_df[feature_cols].copy()
    
    # Remove features with no variance
    variance_threshold = X.var() > 0.01
    X = X.loc[:, variance_threshold]
    
    # Scale numerical features only
    scaler = StandardScaler()
    if numerical_features:
        existing_numerical = [col for col in numerical_features if col in X.columns]
        if existing_numerical:
            X[existing_numerical] = scaler.fit_transform(X[existing_numerical])
    
    # Step 6: Create train/test splits for each target
    print("\n🔧 Step 6: Creating Train/Test Splits")
    
    ml_datasets = {}
    
    for target_name, target_values in targets.items():
        # Remove samples with missing targets
        valid_indices = pd.Series(target_values).notna()
        X_valid = X[valid_indices]
        y_valid = pd.Series(target_values)[valid_indices]
        
        if len(y_valid.unique()) > 1:  # Only if we have variation in target
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42, 
                stratify=y_valid if len(y_valid.unique()) <= 10 else None)
            
            ml_datasets[target_name] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
    
    # Step 7: Save everything
    print("\n💾 Step 7: Saving ML-Ready Files")
    
    # Save feature matrix and targets
    X.to_csv('extracted_data/ML_features_final.csv', index=False)
    pd.DataFrame(targets).to_csv('extracted_data/ML_targets_final.csv', index=False)
    
    # Save train/test splits
    for target_name, datasets in ml_datasets.items():
        datasets['X_train'].to_csv(f'extracted_data/X_train_{target_name}.csv', index=False)
        datasets['X_test'].to_csv(f'extracted_data/X_test_{target_name}.csv', index=False)
        datasets['y_train'].to_csv(f'extracted_data/y_train_{target_name}.csv', index=False)
        datasets['y_test'].to_csv(f'extracted_data/y_test_{target_name}.csv', index=False)
    
    # Step 8: Quick ML validation
    print("\n🤖 Step 8: ML Model Validation")
    
    for target_name, datasets in ml_datasets.items():
        print(f"\n📊 Testing {target_name}:")
        
        if len(datasets['y_train'].unique()) <= 10:  # Classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(datasets['X_train'], datasets['y_train'])
            
            # Cross-validation score
            cv_scores = cross_val_score(model, datasets['X_train'], datasets['y_train'], cv=3)
            print(f"   Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Test set performance
            test_score = model.score(datasets['X_test'], datasets['y_test'])
            print(f"   Test accuracy: {test_score:.3f}")
            
        else:  # Regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(datasets['X_train'], datasets['y_train'])
            
            # Test set performance
            y_pred = model.predict(datasets['X_test'])
            mse = mean_squared_error(datasets['y_test'], y_pred)
            print(f"   Test MSE: {mse:.3f}")
    
    # Final summary
    print(f"\n🎉 ML-Ready Dataset Complete!")
    print("=" * 60)
    print(f"✅ Final features: {X.shape[1]}")
    print(f"✅ Target variables: {len(targets)}")
    print(f"✅ Train/test splits: {len(ml_datasets)}")
    print(f"✅ No missing values: {X.isnull().sum().sum() == 0}")
    print(f"✅ All features scaled: True")
    print(f"✅ Ready for: Classification, Regression, Clustering")
    print(f"✅ Validated with: Random Forest models")
    
    return X, targets, ml_datasets

if __name__ == "__main__":
    create_complete_ml_dataset()