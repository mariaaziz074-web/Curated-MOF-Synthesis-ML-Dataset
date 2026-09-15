# clean_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_clean_plots():
    """Create clean, separate plots"""
    
    # Load the improved dataset
    try:
        df = pd.read_csv('extracted_data/improved_mof_dataset.csv')
    except:
        df = pd.read_csv('extracted_data/mof_cleaned_dataset.csv')
    
    print(f"Creating plots for {len(df)} papers...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Temperature Distribution
    if 'temperature' in df.columns:
        temp_data = df['temperature'].dropna()
        if len(temp_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(temp_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'MOF Synthesis Temperature Distribution\n({len(temp_data)} papers)', fontsize=14)
            plt.xlabel('Temperature (°C)', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)
            plt.axvline(temp_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {temp_data.mean():.1f}°C')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('extracted_data/temperature_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 2. Real Solvents
    if 'solvent' in df.columns:
        solvent_data = df['solvent'].dropna()
        # Filter to real solvents only
        real_solvents = ['DMF', 'DMSO', 'methanol', 'ethanol', 'water', 'acetone', 'THF']
        filtered_solvents = solvent_data[solvent_data.isin(real_solvents)]
        
        if len(filtered_solvents) > 0:
            solvent_counts = filtered_solvents.value_counts().head(10)
            
            plt.figure(figsize=(12, 6))
            solvent_counts.plot(kind='bar', color='lightgreen', alpha=0.8)
            plt.title(f'Most Common Solvents in MOF Synthesis\n({len(filtered_solvents)} papers)', fontsize=14)
            plt.xlabel('Solvents', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('extracted_data/solvent_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 3. pH Distribution
    if 'pH' in df.columns:
        pH_data = df['pH'].dropna()
        # Filter realistic pH values
        pH_data = pH_data[(pH_data >= 0) & (pH_data <= 14)]
        
        if len(pH_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(pH_data, bins=12, alpha=0.7, color='coral', edgecolor='black')
            plt.title(f'pH Distribution in MOF Synthesis\n({len(pH_data)} papers)', fontsize=14)
            plt.xlabel('pH Value', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)
            plt.axvline(pH_data.mean(), color='blue', linestyle='--', 
                       label=f'Mean: {pH_data.mean():.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 14)
            plt.tight_layout()
            plt.savefig('extracted_data/pH_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 4. Metals Distribution
    if 'metal' in df.columns:
        metal_data = df['metal'].dropna()
        # Filter to real metals
        real_metals = ['Zn', 'Cu', 'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Al', 'Zr', 'Ti']
        filtered_metals = metal_data[metal_data.isin(real_metals)]
        
        if len(filtered_metals) > 0:
            metal_counts = filtered_metals.value_counts().head(8)
            
            plt.figure(figsize=(10, 8))
            plt.pie(metal_counts.values, labels=metal_counts.index, autopct='%1.1f%%', 
                   colors=plt.cm.Set3(np.linspace(0, 1, len(metal_counts))))
            plt.title(f'Metal Distribution in MOFs\n({len(filtered_metals)} papers)', fontsize=14)
            plt.tight_layout()
            plt.savefig('extracted_data/metal_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    print("✅ All plots created and saved in extracted_data/ folder!")

if __name__ == "__main__":
    create_clean_plots()