# enhanced_patterns.py
import re
from config import Config

# Additional patterns to improve extraction
enhanced_patterns = {
    'yield': r'(?i)(?:yield|isolated yield)\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
    'reaction_time_hours': r'(?i)(?:stirred|heated|reaction)\s+(?:for\s+)?([0-9]+(?:\.[0-9]+)?)\s*h',
    'molar_ratio': r'(?i)(?:molar ratio|ratio)\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?:[0-9]+(?:\.[0-9]+)?)',
    'concentration': r'(?i)concentration\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:M|mol/L|mmol)',
    'surface_area_bet': r'(?i)BET\s*surface\s*area\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*m²?/g',
    'pore_volume': r'(?i)pore\s*volume\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*cm³?/g',
    'crystal_size': r'(?i)crystal\s*size\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:μm|nm|mm)',
    'heating_rate': r'(?i)heating\s*rate\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*°C/min',
    'gas_flow_rate': r'(?i)(?:gas|flow)\s*rate\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mL/min|L/min)',
}

print("Enhanced patterns that can be added to config.py:")
for key, value in enhanced_patterns.items():
    print(f"'{key}': r'{value}',")

print(f"\nTo use these, add them to Config.DESCRIPTOR_PATTERNS in config.py")
print("Then re-run the extraction to get even more data!")