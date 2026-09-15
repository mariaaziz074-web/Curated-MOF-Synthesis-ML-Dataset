# improved_config.py
import os

class ImprovedConfig:
    # Folder paths
    PDF_FOLDER = "pdf_pdfs"
    OUTPUT_FOLDER = "extracted_data"
    
    # Much more specific and realistic patterns
    DESCRIPTOR_PATTERNS = {
        # Temperature - more specific, realistic ranges
        'temperature': r'(?i)(?:temperature|temp)\s*[:\-=]?\s*([1-9][0-9]{1,3})\s*°?C',
        'synthesis_temperature': r'(?i)(?:synthesis|heated|reaction)\s+(?:at\s+)?([1-9][0-9]{1,2})\s*°?C',
        
        # Time - more specific formats
        'time': r'(?i)(?:for|during|after)\s+([0-9]{1,2})\s*(h|hr|hours?)\b',
        'reaction_time': r'(?i)reaction\s+time[:\s]+([0-9]{1,2})\s*(h|hr|hours?)\b',
        
        # pH - realistic pH range (0-14)
        'pH': r'(?i)pH\s*[=:]?\s*([0-9](?:\.[0-9])?|1[0-4](?:\.[0-9])?)\b',
        
        # Pressure - specific units and ranges
        'pressure': r'(?i)pressure\s*[:\-=]?\s*([0-9]{1,3}(?:\.[0-9])?)\s*(bar|atm|MPa|kPa)\b',
        
        # Real solvents - specific common MOF solvents
        'solvent': r'(?i)\b(DMF|DMSO|methanol|ethanol|water|acetone|THF|toluene|benzene|chloroform|acetonitrile|H2O|MeOH|EtOH)\b',
        
        # Real metals - actual metal ions used in MOFs
        'metal': r'(?i)\b(Zn|Cu|Fe|Co|Ni|Mn|Cr|Al|Zr|Ti|V|Cd|Pb|Ca|Mg|Ba|Sr|La|Ce|Eu|Tb|Dy|Er|Yb|Lu)\b',
        
        # Linkers - common organic linkers
        'linker': r'(?i)\b(BDC|BPDC|NDC|TPDC|BTC|TATB|TCPP|H2BDC|H3BTC|terephthalic|isophthalic|fumaric|succinic)\b',
        
        # Yield - realistic percentage
        'yield': r'(?i)yield\s*[:\-=]?\s*([1-9][0-9]?)\s*%',
        
        # Surface area - realistic BET values
        'surface_area': r'(?i)(?:BET|surface\s+area)\s*[:\-=]?\s*([0-9]{2,4})\s*m²?[/\s]g',
        
        # Pore size - realistic values
        'pore_size': r'(?i)pore\s+size\s*[:\-=]?\s*([0-9]{1,2}(?:\.[0-9])?)\s*(Å|nm)\b',
    }
    
    # Improved section patterns
    SECTION_PATTERNS = {
        'synthesis': r'(?i)synthesis.*?(?=\n\s*[A-Z]{2}|\n\s*\d+\.|$)',
        'experimental': r'(?i)experimental.*?(?=\n\s*[A-Z]{2}|\n\s*\d+\.|$)',
        'results': r'(?i)results.*?(?=\n\s*[A-Z]{2}|\n\s*\d+\.|$)',
        'characterization': r'(?i)characterization.*?(?=\n\s*[A-Z]{2}|\n\s*\d+\.|$)',
    }

# Create output directory
if not os.path.exists(ImprovedConfig.OUTPUT_FOLDER):
    os.makedirs(ImprovedConfig.OUTPUT_FOLDER)