# better_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_realistic_plots():
    """Create plots with realistic MOF data"""
    
    try:
        # Try to load the realistic data first
        df = pd.read_csv('extracted_data/realistic_mof_data.csv')
        print(f"Loaded realistic dataset with {len(df)} papers")
    except:
        print("Realistic data not found, using original data with filters...")
        df = pd.read_csv('extracted_data/mof_cleaned_dataset.csv')
    
    plt.style.use('default')
    
    # 1. Temperature plot (if we have realistic data)
    if 'temperature' in df.columns:
        temp_data = df['temperature'].dropna()
        # Filter to realistic temperatures
        temp_data = temp_data[(temp_data >= 20) & (temp_data <= 500)]
        
        if len(temp_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(temp_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'MOF Synthesis Temperatures\n({len(temp_data)} papers with realistic data)', fontweight='bold')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Number of Papers')
            plt.axvline(temp_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {temp_data.mean():.1f}°C')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('extracted_data/realistic_temperature.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Temperature stats: {temp_data.min():.1f}°C - {temp_data.max():.1f}°C (mean: {temp_data.mean():.1f}°C)")
    
    # 2. Solvents plot
    if 'solvent' in df.columns:
        # Define realistic solvents
        real_solvents = ['DMF', 'DMSO', 'methanol', 'ethanol', 'water', 'acetone', 'THF', 'H2O', 'MeOH', 'EtOH']
        
        solvent_data = df['solvent'].dropna()
        filtered_solvents = solvent_data[solvent_data.isin(real_solvents)]
        
        if len(filtered_solvents) > 0:
            solvent_counts = filtered_solvents.value_counts().head(8)
            
            plt.figure(figsize=(12, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(solvent_counts)))
            bars = plt.bar(range(len(solvent_counts)), solvent_counts.values, 
                          color=colors, alpha=0.8, edgecolor='black')
            plt.title(f'Common Solvents in MOF Synthesis\n({len(filtered_solvents)} papers)', fontweight='bold')
            plt.xlabel('Solvents')
            plt.ylabel('Number of Papers')
            plt.xticks(range(len(solvent_counts)), solvent_counts.index, rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, solvent_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(value), ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('extracted_data/realistic_solvents.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 3. pH plot (if realistic data)
    if 'pH' in df.columns:
        pH_data = df['pH'].dropna()
        # Filter to realistic pH
        pH_data = pH_data[(pH_data >= 0) & (pH_data <= 14)]
        
        if len(pH_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(pH_data, bins=14, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title(f'pH Distribution in MOF Synthesis\n({len(pH_data)} papers)', fontweight='bold')
            plt.xlabel('pH Value')
            plt.ylabel('Number of Papers')
            plt.xlim(0, 14)
            plt.axvline(pH_data.mean(), color='blue', linestyle='--', 
                       label=f'Mean: {pH_data.mean():.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('extracted_data/realistic_pH.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 4. Data completeness
    numeric_cols = [col for col in df.columns if col in ['temperature', 'pH', 'time']]
    if numeric_cols:
        completeness = []
        col_names = []
        
        for col in numeric_cols:
            if col in df.columns:
                count = df[col].count()
                percentage = (count / len(df)) * 100
                completeness.append(percentage)
                col_names.append(f"{col.title()}\n({count} papers)")
        
        if completeness:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(col_names, completeness, color=['skyblue', 'lightgreen', 'orange'], 
                          alpha=0.8, edgecolor='black')
            plt.title('Data Completeness - Realistic Values Only', fontweight='bold')
            plt.ylabel('Completeness (%)')
            plt.ylim(0, 100)
            
            # Add percentage labels
            for bar, perc in zip(bars, completeness):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{perc:.1f}%', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('extracted_data/data_completeness.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    print("✅ Realistic plots created and saved!")

if __name__ == "__main__":
    create_realistic_plots()