# improved_extractor.py
import pandas as pd
import re
from pathlib import Path
from mof_extractor import MOFDescriptorExtractor
from improved_config import ImprovedConfig

class ImprovedMOFExtractor(MOFDescriptorExtractor):
    def __init__(self):
        self.descriptor_patterns = ImprovedConfig.DESCRIPTOR_PATTERNS
        self.section_patterns = ImprovedConfig.SECTION_PATTERNS
    
    def clean_extracted_values(self, df):
        """Clean extracted values to remove outliers and invalid data"""
        
        # Temperature cleaning - realistic range for MOF synthesis (20-500°C)
        if 'temperature' in df.columns:
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
            df.loc[(df['temperature'] < 20) | (df['temperature'] > 500), 'temperature'] = None
        
        # pH cleaning - realistic range (0-14)
        if 'pH' in df.columns:
            df['pH'] = pd.to_numeric(df['pH'], errors='coerce')
            df.loc[(df['pH'] < 0) | (df['pH'] > 14), 'pH'] = None
        
        # Pressure cleaning - realistic range (0.1-50 for bar/atm)
        if 'pressure' in df.columns:
            df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
            df.loc[(df['pressure'] < 0.1) | (df['pressure'] > 100), 'pressure'] = None
        
        # Time cleaning - realistic range (0.1-100 hours)
        if 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df.loc[(df['time'] < 0.1) | (df['time'] > 100), 'time'] = None
        
        return df

def run_improved_extraction():
    """Run improved extraction with better patterns"""
    print("🔧 Running Improved MOF Data Extraction")
    print("=" * 50)
    
    extractor = ImprovedMOFExtractor()
    
    # Process PDFs
    raw_df = extractor.process_pdf_folder()
    
    if raw_df.empty:
        print("No data extracted.")
        return
    
    # Clean the data
    cleaned_df = extractor.clean_extracted_values(raw_df)
    
    # Save improved data
    cleaned_df.to_csv('extracted_data/improved_mof_dataset.csv', index=False)
    cleaned_df.to_excel('extracted_data/improved_mof_dataset.xlsx', index=False)
    
    print(f"✅ Improved extraction complete!")
    print(f"📊 Dataset shape: {cleaned_df.shape}")
    
    # Show improved statistics
    for col in ['temperature', 'pH', 'time', 'pressure']:
        if col in cleaned_df.columns:
            valid_data = cleaned_df[col].dropna()
            if len(valid_data) > 0:
                print(f"\n{col.upper()}:")
                print(f"  Papers with data: {len(valid_data)}")
                print(f"  Range: {valid_data.min():.1f} - {valid_data.max():.1f}")
                print(f"  Mean: {valid_data.mean():.1f}")

if __name__ == "__main__":
    run_improved_extraction()