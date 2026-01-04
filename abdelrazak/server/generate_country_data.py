"""
One-time script to generate country_data.json from Climate_Data_Yearly_Final.csv
Run this script whenever the source CSV is updated.
"""
import pandas as pd
import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent  # DALAS-Project root
OUTPUT_PATH = SCRIPT_DIR / "country_data.json"

def generate_country_data():
    """Extract latest year data for each country and save as JSON."""
    climate_path = DATA_DIR / "Climate_Data_Yearly_Final.csv"
    
    print(f"Reading CSV from: {climate_path}")
    df = pd.read_csv(climate_path)
    
    # Get most recent year's data for each country
    latest = df.groupby('country_name')['year'].max().reset_index()
    df = df.merge(latest, on=['country_name', 'year'])
    
    # Convert to dictionary indexed by country name
    country_data = df.set_index('country_name').to_dict('index')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    country_data = convert_types(country_data)
    
    # Save to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(country_data, f, indent=2)
    
    print(f"✓ Generated {OUTPUT_PATH}")
    print(f"✓ Saved data for {len(country_data)} countries")

if __name__ == "__main__":
    generate_country_data()

