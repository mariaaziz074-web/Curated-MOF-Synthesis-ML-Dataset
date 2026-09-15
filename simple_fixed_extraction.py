# simple_fixed_extraction.py
import pandas as pd
import re
import pdfplumber
from pathlib import Path

def extract_realistic_mof_data():
    """Extract MOF data with realistic patterns"""
    
    # Much better, more specific patterns
    patterns = {
        # Temperature - realistic synthesis temperatures (50-400°C)
        'temperature': r'(?i)(?:at|temperature|heated to)\s+([1-4]?[0-9]{2})\s*°?C\b',
        
        # Common MOF solvents
        'solvent': r'(?i)\b(DMF|DMSO|methanol|ethanol|water|acetone|THF|toluene|H2O|MeOH|EtOH|acetonitrile|dichloromethane|chloroform)\b',
        
        # Real metal ions used in MOFs
        'metal': r'(?i)\b(Zn|Cu|Fe|Co|Ni|Mn|Cr|Al|Zr|Ti|V|Cd|Pb|Ca|Mg|Ba|Sr|La|Ce|Eu|Tb|Dy|Er|Yb|Lu)(?:\s*\(?II?\)?)?\b',
        
        # pH - realistic range
        'pH': r'(?i)pH\s*[=:\s]\s*([0-9](?:\.[0-9])?|1[0-4](?:\.[0-9])?)\b',
        
        # Time - realistic reaction times
        'time': r'(?i)(?:for|after|during)\s+([1-9][0-9]?)\s*(h|hr|hours?|min|minutes?)\b',
        
        # Yield - realistic percentages
        'yield': r'(?i)yield\s*[:\-=]?\s*([1-9][0-9]?)\s*%',
        
        # Common organic linkers
        'linker': r'(?i)\b(H2BDC|H3BTC|BDC|BTC|terephthalic|isophthalic|fumaric|succinic|adipic|2-aminoterephthalic|BPDC|NDC|TPDC)\b',
    }
    
    pdf_folder = Path("pdf_pdfs")
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"Processing {len(pdf_files)} PDF files with improved patterns...")
    
    all_data = []
    
    for i, pdf_file in enumerate(pdf_files[:10], 1):  # Process first 10 files for testing
        print(f"Processing {i}/10: {pdf_file.name}")
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                for page in pdf.pages[:5]:  # First 5 pages only
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + " "
                
                if not full_text:
                    continue
                
                # Extract data for this paper
                paper_data = {'filename': pdf_file.name}
                
                for desc_type, pattern in patterns.items():
                    matches = re.findall(pattern, full_text, re.IGNORECASE)
                    
                    if matches:
                        if desc_type in ['temperature', 'pH', 'time']:
                            # Take first numeric value
                            try:
                                value = float(matches[0]) if isinstance(matches[0], str) else float(matches[0][0])
                                # Apply realistic filters
                                if desc_type == 'temperature' and (value < 20 or value > 500):
                                    continue
                                if desc_type == 'pH' and (value < 0 or value > 14):
                                    continue
                                if desc_type == 'time' and (value < 0.1 or value > 100):
                                    continue
                                paper_data[desc_type] = value
                            except:
                                pass
                        else:
                            # Take most common value
                            if matches:
                                paper_data[desc_type] = max(set(matches), key=matches.count)
                
                if len(paper_data) > 1:  # If we found at least one descriptor
                    all_data.append(paper_data)
                    
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    if not all_data:
        print("No valid data extracted!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save results
    df.to_csv('extracted_data/realistic_mof_data.csv', index=False)
    df.to_excel('extracted_data/realistic_mof_data.xlsx', index=False)
    
    # Show summary
    print(f"\n🎉 Extracted realistic data from {len(df)} papers!")
    print(f"📊 Dataset shape: {df.shape}")
    
    for col in df.columns:
        if col != 'filename':
            valid_count = df[col].count()
            if valid_count > 0:
                print(f"  {col}: {valid_count} papers")
                if col in ['temperature', 'pH', 'time']:
                    values = df[col].dropna()
                    print(f"    Range: {values.min():.1f} - {values.max():.1f}")
                    print(f"    Mean: {values.mean():.1f}")
                else:
                    top_values = df[col].value_counts().head(3)
                    print(f"    Top values: {list(top_values.index)}")
    
    return df

if __name__ == "__main__":
    extract_realistic_mof_data()