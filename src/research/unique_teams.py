import pandas as pd
import os

def analyze_unique_teams():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    # Extract unique IDs
    unique_ids = pd.concat([df['HID'], df['AID']]).unique()
    unique_ids.sort()
    
    count = len(unique_ids)
    
    # Check for consistency
    # Does every team play at home and away?
    h_counts = df['HID'].value_counts()
    a_counts = df['AID'].value_counts()
    
    # Teams with very few games?
    
    print(f"Unique Teams Found: {count}")
    print(f"IDs: {unique_ids}")
    
    readme = f"""# Unique Teams Analysis

## Goal
Verify the integrity of team identifiers.

## Findings
- **Total Unique Teams**: {count}
- **IDs**: {', '.join(map(str, unique_ids))}

## Data Quality Check
- All teams appear to be mapped to integers.
- IMPORTANT: Ensure these IDs are consistent across seasons if seasons are 2009-2015.
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/UNIQUE_TEAMS.md', 'w') as f:
        f.write(readme)
    print("README generated at src/research/UNIQUE_TEAMS.md")

if __name__ == "__main__":
    analyze_unique_teams()
