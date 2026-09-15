# utils.py
import re
import pandas as pd
from pathlib import Path

def clean_extracted_text(text):
    """Clean and preprocess extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s°\-\.\,\:\;\(\)\[\]\/\%\+\=]', '', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    return text.strip()

def extract_multiple_values(text, pattern):
    """Extract multiple values for a descriptor"""
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    if matches:
        # Clean and convert to appropriate type
        cleaned_matches = []
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1] if len(match) > 1 else ""
            
            # Try to convert to float if it's a number
            try:
                cleaned_matches.append(float(match))
            except ValueError:
                cleaned_matches.append(str(match).strip())
        
        return cleaned_matches
    return []

def save_to_multiple_formats(df, base_filename):
    """Save dataframe to multiple formats"""
    from config import Config
    
    # CSV
    csv_path = Path(Config.OUTPUT_FOLDER) / f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Excel
    excel_path = Path(Config.OUTPUT_FOLDER) / f"{base_filename}.xlsx"
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel: {excel_path}")
    
    # JSON for complex data
    json_path = Path(Config.OUTPUT_FOLDER) / f"{base_filename}.json"
    df.to_json(json_path, indent=2, orient='records')
    print(f"Saved JSON: {json_path}")
    
    return csv_path, excel_path, json_path

def create_summary_stats(df):
    """Create summary statistics of extracted data"""
    summary = {
        'total_papers': len(df),
        'papers_with_temperature': len(df[df['temperature'].notna()]) if 'temperature' in df.columns else 0,
        'papers_with_time': len(df[df['time'].notna()]) if 'time' in df.columns else 0,
        'papers_with_pH': len(df[df['pH'].notna()]) if 'pH' in df.columns else 0,
        'papers_with_yield': len(df[df['yield'].notna()]) if 'yield' in df.columns else 0,
        'unique_solvents': df['solvent'].nunique() if 'solvent' in df.columns else 0,
        'unique_metals': df['metal'].nunique() if 'metal' in df.columns else 0,
    }
    
    return summary