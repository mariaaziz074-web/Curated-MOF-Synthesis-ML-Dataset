# comprehensive_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_comprehensive_analysis():
    """Create comprehensive analysis with all realistic data"""
    
    # Load the full dataset
    try:
        df = pd.read_csv('extracted_data/full_realistic_mof_data.csv')
        print(f"📊 Loaded full dataset with {len(df)} papers and {df.shape[1]} descriptors")
    except:
        print("❌ Full dataset not found. Run full_realistic_extraction.py first")
        return
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Temperature distributions (multiple subplots)
    ax1 = plt.subplot(3, 4, 1)
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    for col in temp_cols:
        if col in df.columns:
            temp_data = df[col].dropna()
            if len(temp_data) > 0:
                ax1.hist(temp_data, alpha=0.6, label=col, bins=10)
    ax1.set_title(f'Temperature Distributions\n({sum([df[col].count() for col in temp_cols])} total data points)')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Solvent distribution
    ax2 = plt.subplot(3, 4, 2)
    if 'solvent' in df.columns:
        solvent_data = df['solvent'].dropna()
        if len(solvent_data) > 0:
            solvent_counts = solvent_data.value_counts().head(8)
            ax2.bar(range(len(solvent_counts)), solvent_counts.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(solvent_counts))))
            ax2.set_title(f'Solvents ({len(solvent_data)} papers)')
            ax2.set_xticks(range(len(solvent_counts)))
            ax2.set_xticklabels(solvent_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Count')
    
    # 3. Metal distribution (pie chart)
    ax3 = plt.subplot(3, 4, 3)
    if 'metal' in df.columns:
        metal_data = df['metal'].dropna()
        if len(metal_data) > 0:
            metal_counts = metal_data.value_counts().head(8)
            ax3.pie(metal_counts.values, labels=metal_counts.index, autopct='%1.1f%%')
            ax3.set_title(f'Metals ({len(metal_data)} papers)')
    
    # 4. pH distribution
    ax4 = plt.subplot(3, 4, 4)
    if 'pH' in df.columns:
        pH_data = df['pH'].dropna()
        if len(pH_data) > 0:
            ax4.hist(pH_data, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_title(f'pH Values ({len(pH_data)} papers)')
            ax4.set_xlabel('pH')
            ax4.set_ylabel('Count')
            ax4.set_xlim(0, 14)
    
    # 5. Time analysis
    ax5 = plt.subplot(3, 4, 5)
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    for col in time_cols:
        if col in df.columns:
            time_data = df[col].dropna()
            if len(time_data) > 0:
                ax5.hist(time_data, alpha=0.6, label=col.replace('_', ' '), bins=8)
    ax5.set_title('Reaction Times')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Count')
    ax5.legend()
    
    # 6. Yield distribution
    ax6 = plt.subplot(3, 4, 6)
    if 'yield' in df.columns:
        yield_data = df['yield'].dropna()
        if len(yield_data) > 0:
            ax6.hist(yield_data, bins=10, alpha=0.7, color='gold', edgecolor='black')
            ax6.set_title(f'Yield Distribution ({len(yield_data)} papers)')
            ax6.set_xlabel('Yield (%)')
            ax6.set_ylabel('Count')
    
    # 7. Linker types
    ax7 = plt.subplot(3, 4, 7)
    if 'linker' in df.columns:
        linker_data = df['linker'].dropna()
        if len(linker_data) > 0:
            linker_counts = linker_data.value_counts().head(6)
            ax7.barh(range(len(linker_counts)), linker_counts.values, color='lightblue')
            ax7.set_title(f'Linkers ({len(linker_data)} papers)')
            ax7.set_yticks(range(len(linker_counts)))
            ax7.set_yticklabels(linker_counts.index)
            ax7.set_xlabel('Count')
    
    # 8. Surface area
    ax8 = plt.subplot(3, 4, 8)
    if 'surface_area' in df.columns:
        sa_data = df['surface_area'].dropna()
        if len(sa_data) > 0:
            ax8.hist(sa_data, bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
            ax8.set_title(f'Surface Area ({len(sa_data)} papers)')
            ax8.set_xlabel('Surface Area (m²/g)')
            ax8.set_ylabel('Count')
    
    # 9. Data completeness matrix
    ax9 = plt.subplot(3, 4, 9)
    numeric_cols = [col for col in df.columns if col != 'filename']
    completeness = [df[col].count() for col in numeric_cols]
    ax9.bar(range(len(completeness)), completeness, color='orange', alpha=0.7)
    ax9.set_title('Data Completeness')
    ax9.set_xticks(range(len(numeric_cols)))
    ax9.set_xticklabels([col[:8] for col in numeric_cols], rotation=45, ha='right')
    ax9.set_ylabel('Papers with Data')
    
    # 10. Temperature vs pH correlation (if both exist)
    ax10 = plt.subplot(3, 4, 10)
    if 'temperature' in df.columns and 'pH' in df.columns:
        temp_data = df['temperature'].dropna()
        ph_data = df['pH'].dropna()
        common_indices = df.index[df['temperature'].notna() & df['pH'].notna()]
        if len(common_indices) > 0:
            ax10.scatter(df.loc[common_indices, 'temperature'], 
                        df.loc[common_indices, 'pH'], 
                        alpha=0.6, color='purple')
            ax10.set_title('Temperature vs pH')
            ax10.set_xlabel('Temperature (°C)')
            ax10.set_ylabel('pH')
    
    # 11. Summary statistics table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    summary_text = f"""
    DATASET SUMMARY
    ===============
    Total Papers: {len(df)}
    
    Data Points Found:
    • Temperature: {df['temperature'].count() if 'temperature' in df.columns else 0}
    • Solvents: {df['solvent'].count() if 'solvent' in df.columns else 0}
    • Metals: {df['metal'].count() if 'metal' in df.columns else 0}
    • pH Values: {df['pH'].count() if 'pH' in df.columns else 0}
    • Yields: {df['yield'].count() if 'yield' in df.columns else 0}
    • Linkers: {df['linker'].count() if 'linker' in df.columns else 0}
    
    Extraction Success: {(len(df)/54)*100:.1f}%
    """
    ax11.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    # 12. Most productive research areas
    ax12 = plt.subplot(3, 4, 12)
    # Count descriptors per paper
    descriptor_counts = []
    for idx, row in df.iterrows():
        count = row.notna().sum() - 1  # exclude filename
        descriptor_counts.append(count)
    
    ax12.hist(descriptor_counts, bins=8, alpha=0.7, color='red', edgecolor='black')
    ax12.set_title('Descriptors per Paper')
    ax12.set_xlabel('Number of Descriptors Found')
    ax12.set_ylabel('Number of Papers')
    
    plt.suptitle('Comprehensive MOF Literature Analysis - 54 Research Papers', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('extracted_data/comprehensive_mof_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comprehensive analysis complete!")
    print("💾 Saved as: comprehensive_mof_analysis.png")

if __name__ == "__main__":
    create_comprehensive_analysis()