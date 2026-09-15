# fixed_mof_extractor.py
import os
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from config import Config
from utils import clean_extracted_text, extract_multiple_values, save_to_multiple_formats

class MOFDescriptorExtractor:
    def __init__(self):
        self.descriptor_patterns = Config.DESCRIPTOR_PATTERNS
        self.section_patterns = Config.SECTION_PATTERNS
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with structure preservation"""
        text_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = clean_extracted_text(page_text)
                        text_content.append({
                            'page': page_num + 1,
                            'text': cleaned_text
                        })
                        
            print(f"Successfully extracted {len(text_content)} pages from {pdf_path.name}")
            return text_content
            
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return []
    
    def extract_descriptors_from_text(self, text):
        """Extract descriptors from text using patterns"""
        descriptors = {}
        
        for descriptor_type, pattern in self.descriptor_patterns.items():
            values = extract_multiple_values(text, pattern)
            
            if values:
                if len(values) == 1:
                    descriptors[descriptor_type] = values[0]
                else:
                    descriptors[descriptor_type] = values[0]
                    descriptors[f"{descriptor_type}_all"] = values
        
        return descriptors
    
    def parse_sections(self, text):
        """Parse text into sections"""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                sections[section_name] = matches[0][:1000]
        
        return sections
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file"""
        text_content = self.extract_text_from_pdf(pdf_path)
        
        if not text_content:
            return None
        
        full_text = "\n".join([page['text'] for page in text_content])
        
        paper_data = {
            'filename': pdf_path.name,
            'total_pages': len(text_content),
            'total_characters': len(full_text)
        }
        
        overall_descriptors = self.extract_descriptors_from_text(full_text)
        paper_data.update(overall_descriptors)
        
        sections = self.parse_sections(full_text)
        
        for section_name, section_text in sections.items():
            section_descriptors = self.extract_descriptors_from_text(section_text)
            
            for desc_type, desc_value in section_descriptors.items():
                section_key = f"{section_name}_{desc_type}"
                paper_data[section_key] = desc_value
        
        for section_name in self.section_patterns.keys():
            paper_data[f"has_{section_name}_section"] = section_name in sections
        
        return paper_data
    
    def process_pdf_folder(self, folder_path=None):
        """Process all PDFs in the specified folder"""
        if folder_path is None:
            folder_path = Config.PDF_FOLDER
        
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Folder {folder_path} does not exist!")
            return pd.DataFrame()
        
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return pd.DataFrame()
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        all_data = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            paper_data = self.process_single_pdf(pdf_file)
            
            if paper_data:
                all_data.append(paper_data)
            else:
                print(f"Failed to process {pdf_file.name}")
        
        if not all_data:
            print("No data extracted from any PDF!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        print(f"\nSuccessfully processed {len(df)} papers")
        print(f"Extracted columns: {list(df.columns)}")
        
        return df
    
    def clean_and_prepare_dataset(self, df):
        """Clean and prepare the dataset for ML"""
        if df.empty:
            return df
        
        numeric_columns = []
        categorical_columns = []
        
        for col in df.columns:
            if col in ['filename', 'total_pages', 'total_characters'] or col.startswith('has_'):
                continue
                
            numeric_count = 0
            for val in df[col].dropna():
                try:
                    float(val)
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(df[col].dropna()) * 0.7:
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Numeric columns: {numeric_columns}")
        print(f"Categorical columns: {categorical_columns}")
        
        return df