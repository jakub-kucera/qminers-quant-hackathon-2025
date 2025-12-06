import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

def analyze_coef_extrapolation():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    # We want to see how the importance of a feature (e.g., Goal Difference) changes over seasons
    # Rolling Regression
    
    df['Diff'] = df['HS'] - df['AS'] if 'HS' in df.columns else df['HSC'] - df['ASC']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    coefs = []
    dates = []
    
    window = 500
    step = 100
    
    # Feature: Previous Game Goal Diff (Simple momentum feature)
    # Just to test coefficient stability.
    # We need a Lagged feature.
    
    last_diff = {}
    prev_diffs = []
    
    for i, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        
        ph = last_diff.get(hid, 0)
        pa = last_diff.get(aid, 0)
        
        prev_diffs.append(ph - pa) # Feature: Home Prev Diff - Away Prev Diff
        
        last_diff[hid] = row['Diff']
        last_diff[aid] = -row['Diff']
        
    df['Feat_PrevDiff'] = prev_diffs
    
    # Rolling Regression
    # Target: Margin (Diff)
    
    for start in range(0, len(df) - window, step):
        subset = df.iloc[start : start+window]
        X = subset[['Feat_PrevDiff']].values
        y = subset['Diff'].values
        
        if len(X) < window: continue
        
        model = LinearRegression()
        model.fit(X, y)
        
        coefs.append(model.coef_[0])
        dates.append(subset['Date'].iloc[-1])
        
    plt.figure(figsize=(10, 5))
    plt.plot(dates, coefs)
    plt.title('Rolling Coefficient of "Prev Goal Diff" (Window=500)')
    plt.ylabel('Coefficient Value')
    plt.xlabel('Date')
    plt.grid()
    plt.savefig('src/research/coef_extrapolation.png')
    
    std = np.std(coefs)
    mean = np.mean(coefs)
    
    readme = f"""# Coefficient Extrapolation (Stability)

## Goal
Check if the predictive power of a feature (Previous Goal Difference) assumes a constant value or drifts over time.

## Findings
- **Mean Coefficient**: {mean:.4f}
- **Std Dev**: {std:.4f}
- **Stability**: See `coef_extrapolation.png`.

## Implication
- If the plot is flat, the game meta is stable.
- If it trends up/down, the importance of "Momentum" is changing.
- Large fluctuations suggest high variance or regime changes (e.g., rule changes).
"""

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/COEF_EXTRAP_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_coef_extrapolation()
