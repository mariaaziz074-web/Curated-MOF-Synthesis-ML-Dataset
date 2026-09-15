# full_realistic_extraction.py
import pandas as pd
import re
import pdfplumber
from pathlib import Path

def extract_all_papers():
    """Extract MOF data from ALL papers with improved patterns"""
    
    # Enhanced patterns with better specificity
    patterns = {
        # Temperature - more contexts
        'temperature': r'(?i)(?:heated|temperature|at)\s+([1-4]?[0-9]{2})\s*°?C\b',
        'synthesis_temp': r'(?i)(?:synthesis|heated to|reaction at)\s+([5-9][0-9]|[1-4][0-9]{2})\s*°?C',
        
        # Common MOF solvents - more comprehensive
        'solvent': r'(?i)\b(DMF|DMSO|methanol|ethanol|water|acetone|THF|toluene|H2O|MeOH|EtOH|acetonitrile|dichloromethane|chloroform|diethylformamide|dimethylsulfoxide|tetrahydrofuran)\b',
        
        # Metal ions - more specific patterns
        'metal': r'(?i)\b(Zn|Cu|Fe|Co|Ni|Mn|Cr|Al|Zr|Ti|V|Cd|Pb|Ca|Mg|Ba|Sr|La|Ce|Eu|Tb|Dy|Er|Yb|Lu)(?:\s*\(?II?\)?|[\s\-]\w+)?\b',
        
        # pH values
        'pH': r'(?i)pH\s*[=:\s]\s*([0-9](?:\.[0-9])?|1[0-4](?:\.[0-9])?)\b',
        
        # Time - multiple formats
        'time_hours': r'(?i)(?:for|after|during)\s+([1-9][0-9]?)\s*(?:h|hr|hours?)\b',
        'time_minutes': r'(?i)(?:for|after|during)\s+([1-9][0-9]?)\s*(?:min|minutes?)\b',
        
        # Yield
        'yield': r'(?i)yield\s*[:\-=]?\s*([1-9][0-9]?)\s*%',
        
        # Linkers - comprehensive list
        'linker': r'(?i)\b(H2BDC|H3BTC|BDC|BTC|terephthalic|isophthalic|fumaric|succinic|adipic|2-aminoterephthalic|BPDC|NDC|TPDC|phthalic|benzoic|trimesic)\b',
        
        # Surface area
        'surface_area': r'(?i)(?:BET|surface\s+area)\s*[:\-=]?\s*([0-9]{2,4})\s*m²?[/\s]?g',
        
        # Pressure
        'pressure': r'(?i)pressure\s*[:\-=]?\s*([0-9]{1,2}(?:\.[0-9])?)\s*(bar|atm|MPa)\b',
    }
    
    pdf_folder = Path("pdf_pdfs")
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"🔬 Processing ALL {len(pdf_files)} PDF files with enhanced patterns...")
    
    all_data = []
    successful_extractions = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name[:50]}...")
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                # Process first 10 pages to capture most relevant content
                for page in pdf.pages[:10]:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + " "
                
                if not full_text:
                    continue
                
                # Extract data for this paper
                paper_data = {'filename': pdf_file.name}
                found_any_data = False
                
                for desc_type, pattern in patterns.items():
                    matches = re.findall(pattern, full_text, re.IGNORECASE)
                    
                    if matches:
                        if desc_type in ['temperature', 'synthesis_temp', 'pH', 'time_hours', 'time_minutes', 'yield', 'surface_area', 'pressure']:
                            # Take first valid numeric value
                            try:
                                if isinstance(matches[0], tuple):
                                    value = float(matches[0][0]) if matches[0][0] else float(matches[0][1])
                                else:
                                    value = float(matches[0])
                                
                                # Apply realistic filters
                                if desc_type in ['temperature', 'synthesis_temp'] and (value < 20 or value > 600):
                                    continue
                                if desc_type == 'pH' and (value < 0 or value > 14):
                                    continue
                                if desc_type in ['time_hours'] and (value < 0.1 or value > 200):
                                    continue
                                if desc_type == 'yield' and (value < 1 or value > 99):
                                    continue
                                if desc_type == 'surface_area' and (value < 10 or value > 5000):
                                    continue
                                if desc_type == 'pressure' and (value < 0.1 or value > 100):
                                    continue
                                
                                paper_data[desc_type] = value
                                found_any_data = True
                            except:
                                pass
                        else:
                            # For text values (solvent, metal, linker)
                            if matches:
                                # Take most common value or first if all unique
                                unique_matches = list(set([m.lower() if isinstance(m, str) else m for m in matches]))
                                paper_data[desc_type] = unique_matches[0]
                                found_any_data = True
                
                if found_any_data:
                    all_data.append(paper_data)
                    successful_extractions += 1
                    
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}...")
    
    if not all_data:
        print("No valid data extracted!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Combine time columns
    if 'time_hours' in df.columns and 'time_minutes' in df.columns:
        df['time_total_hours'] = df['time_hours'].fillna(0) + (df['time_minutes'].fillna(0) / 60)
        df['time_total_hours'] = df['time_total_hours'].replace(0, None)
    
    # Save results
    df.to_csv('extracted_data/full_realistic_mof_data.csv', index=False)
    df.to_excel('extracted_data/full_realistic_mof_data.xlsx', index=False)
    
    # Comprehensive summary
    print(f"\n🎉 Successfully extracted data from {successful_extractions}/{len(pdf_files)} papers!")
    print(f"📊 Final dataset shape: {df.shape}")
    print(f"💾 Files saved: full_realistic_mof_data.csv/xlsx")
    
    print(f"\n📈 Data Summary:")
    print("-" * 50)
    
    for col in sorted(df.columns):
        if col != 'filename':
            valid_count = df[col].count()
            if valid_count > 0:
                percentage = (valid_count / len(df)) * 100
                print(f"  {col:20s}: {valid_count:2d} papers ({percentage:4.1f}%)")
                
                if col in ['temperature', 'synthesis_temp', 'pH', 'time_hours', 'time_minutes', 'yield', 'surface_area', 'pressure', 'time_total_hours']:
                    values = df[col].dropna()
                    print(f"                         Range: {values.min():.1f} - {values.max():.1f}")
                    print(f"                         Mean: {values.mean():.1f}")
                else:
                    top_values = df[col].value_counts().head(3)
                    print(f"                         Top: {', '.join(top_values.index)}")
                print()
    
    return df

if __name__ == "__main__":
    extract_all_papers()