# run_extraction.py
import pandas as pd
from pathlib import Path
from mof_extractor import MOFDescriptorExtractor
from config import Config
from utils import save_to_multiple_formats, create_summary_stats

def main():
    """Main execution function"""
    print("=== MOF PDF Data Extraction Tool ===")
    print(f"Looking for PDFs in: {Config.PDF_FOLDER}")
    
    # Initialize extractor
    extractor = MOFDescriptorExtractor()
    
    # Process PDFs
    raw_df = extractor.process_pdf_folder()
    
    if raw_df.empty:
        print("No data extracted. Please check your PDF folder and files.")
        return
    
    # Save raw data
    print("\n=== Saving Raw Data ===")
    save_to_multiple_formats(raw_df, "mof_raw_extracted_data")
    
    # Clean and prepare dataset
    print("\n=== Cleaning and Preparing Dataset ===")
    clean_df = extractor.clean_and_prepare_dataset(raw_df.copy())
    
    # Save cleaned data
    save_to_multiple_formats(clean_df, "mof_cleaned_dataset")
    
    # Create summary statistics
    summary = create_summary_stats(clean_df)
    
    print("\n=== Extraction Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    save_to_multiple_formats(summary_df, "extraction_summary")
    
    # Display sample data
    print("\n=== Sample Extracted Data ===")
    print(clean_df.head())
    
    print(f"\n=== Extraction Complete ===")
    print(f"Files saved in: {Config.OUTPUT_FOLDER}/")
    print("- mof_raw_extracted_data.csv/xlsx/json")
    print("- mof_cleaned_dataset.csv/xlsx/json") 
    print("- extraction_summary.csv/xlsx/json")

if __name__ == "__main__":
    main()