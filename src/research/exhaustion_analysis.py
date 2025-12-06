import pandas as pd
import numpy as np
import os

def analyze_exhaustion():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate Rest Days
    last_game = {} # TeamID -> Date
    
    rest_home = []
    rest_away = []
    
    for i, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        date = row['Date']
        
        # Home Rest
        if hid in last_game:
            days_h = (date - last_game[hid]).days
        else:
            days_h = 7 # Assume rested if new season/start
            
        # Away Rest
        if aid in last_game:
            days_a = (date - last_game[aid]).days
        else:
            days_a = 7
            
        rest_home.append(days_h)
        rest_away.append(days_a)
        
        last_game[hid] = date
        last_game[aid] = date
        
    df['RestH'] = rest_home
    df['RestA'] = rest_away
    
    # Cap rest > 5 days (Long rest vs Short rest)
    df['RestH_Cat'] = df['RestH'].clip(1, 5)
    df['RestA_Cat'] = df['RestA'].clip(1, 5)
    
    # Analyze Win Rate by Rest Config
    # Case: Home Tired (1 day) vs Away Rested (3+ days)
    tired_home = df[(df['RestH'] == 1) & (df['RestA'] >= 3)]
    rested_home = df[(df['RestH'] >= 3) & (df['RestA'] == 1)]
    
    print(f"Tired Home (Home 1d, Away 3+d) Games: {len(tired_home)}")
    win_rate_tired = tired_home['H'].mean()
    
    print(f"Rested Home (Home 3+d, Away 1d) Games: {len(rested_home)}")
    win_rate_rested = rested_home['H'].mean()
    
    # Baseline
    avg_win = df['H'].mean()
    
    readme = f"""# Exhaustion / Rest Analysis

## Goal
Quantify the impact of "Back-to-Back" games (Rest = 1 day) vs Rested teams (Rest >= 3 days).

## Findings
- **Baseline Home Win Rate**: {avg_win:.2%}
- **Tired Home (Rest=1 vs Rest>=3)**: {win_rate_tired:.2%} (Difference: {win_rate_tired - avg_win:.2%})
- **Rested Home (Rest>=3 vs Rest=1)**: {win_rate_rested:.2%} (Difference: {win_rate_rested - avg_win:.2%})

## Implication
- "Schedule Loss": Does playing yesterday significantly hurt chances? 
- If the difference is > 2-3%, `RestDays` MUST be a feature in the main model.
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/EXHAUSTION_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_exhaustion()
