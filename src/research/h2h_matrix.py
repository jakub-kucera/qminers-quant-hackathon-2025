import pandas as pd
import numpy as np
import os

def analyze_h2h_matrix():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    df['Expectation'] = 1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA'])
    
    # H2H Key
    # We want to see if specific matchups consistently defy odds
    # e.g., Team A always beats Team B despite odds
    
    h2h_stats = {}
    
    for i, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        key = tuple(sorted([hid, aid])) # Unique Matchup Key
        
        if key not in h2h_stats:
            h2h_stats[key] = {'exp': 0, 'actual': 0, 'games': 0, 'pnl': 0}
            
        # Orientation
        # If key[0] is home, add normal. If key[1] is home, we need to flip relative to Key[0]
        # Let's track absolute "Home Overperformance" for simplicity logic
        
        # Simpler: Just Track (HID, AID) specific directional bias?
        # But games are rare (maybe 2-4 per season).
        # Let's stick to the ported logic: Aggregate per matchup key.
        
        bias = row['H'] - row['Expectation']
        
        h2h_stats[key]['exp'] += row['Expectation']
        h2h_stats[key]['actual'] += row['H']
        h2h_stats[key]['games'] += 1
        
    # Analyze significant deviations
    results = []
    
    for k, v in h2h_stats.items():
        if v['games'] >= 10: # Only frequent matchups
            bias = (v['actual'] - v['exp']) / v['games']
            results.append([k, v['games'], bias])
            
    res_df = pd.DataFrame(results, columns=['Matchup', 'Games', 'Avg_Bias']).sort_values('Avg_Bias')
    
    top_bias = res_df.tail(5)
    bot_bias = res_df.head(5)
    
    readme = f"""# Head-to-Head (H2H) Tendencies

## Goal
Identify "Bogey Teams" or matchups where one team consistently overperforms the odds.

## Findings
filtered for Matchups with >= 10 games.

### Top Home Overperformers (Home Team wins more than expected)
```
{top_bias.to_string(index=False)}
```

### Top Away Overperformers (Home Team underperforms)
```
{bot_bias.to_string(index=False)}
```

## Implication
- If significant biases exist (e.g., > 10-20% edge), adding an `H2H_Bias` feature or `MatchupID` embedding is valuable.
- Caution: Small sample sizes (even 10 games is small for variance).
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/H2H_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_h2h_matrix()
