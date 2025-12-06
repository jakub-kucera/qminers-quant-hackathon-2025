import pandas as pd
import numpy as np
import os

def analyze_team_stationarity():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    def inverse_sigmoid(x):
        return np.log(x / (1 - x + 1e-9))

    # Implied prob (normalized)
    imp_h = 1/df['OddsH']
    imp_a = 1/df['OddsA']
    total = imp_h + imp_a + (1/df['OddsD'] if 'OddsD' in df.columns else 0)
    
    df['ProbH'] = imp_h / total
    df['ProbA'] = imp_a / total
    
    # Market Perceived Strength (Log Odds)
    df['StrengthH'] = inverse_sigmoid(df['ProbH'])
    df['StrengthA'] = inverse_sigmoid(df['ProbA']) # Actually Strength A is approx -StrengthH if zero sum
    
    # Avg Strength per Season
    h_group = df.groupby(['Season', 'HID'])['StrengthH'].mean().reset_index().rename(columns={'HID': 'TeamID', 'StrengthH': 'AvgStrength'})
    a_group = df.groupby(['Season', 'AID'])['StrengthA'].mean().reset_index().rename(columns={'AID': 'TeamID', 'StrengthA': 'AvgStrength'})
    
    combined = pd.concat([h_group, a_group]).groupby(['Season', 'TeamID']).mean().reset_index()
    
    # Check correlation between seasons
    # Does strength persist?
    seasons = sorted(combined['Season'].unique())
    correlations = []
    
    prev_season = None
    for s in seasons:
        curr = combined[combined['Season'] == s].set_index('TeamID')
        if prev_season is not None:
            prev = combined[combined['Season'] == prev_season].set_index('TeamID')
            # Join
            joined = curr.join(prev, lsuffix='_curr', rsuffix='_prev').dropna()
            corr = joined['AvgStrength_curr'].corr(joined['AvgStrength_prev'])
            correlations.append((s, corr))
        prev_season = s
        
    avg_corr = np.mean([c[1] for c in correlations])
    
    readme = f"""# Team Strength Stationarity

## Goal
Determine if "Good Teams stay Good" year-over-year.

## Findings
- **Year-over-Year Correlation**: {avg_corr:.4f}

## Results by Season Pair
{pd.DataFrame(correlations, columns=['Season', 'Correlation']).to_string(index=False)}

## Implication
- High correlation (> 0.5) means we can rely on last year's data (Model `start_season` can include old data).
- Low correlation means roster turnover is high, and looking back too far adds noise.
"""
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/TEAM_STATIONARITY_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_team_stationarity()
