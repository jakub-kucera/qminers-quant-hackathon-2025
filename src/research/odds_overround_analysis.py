import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

def analyze_odds_overround():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    df = df.dropna(subset=['OddsH', 'OddsA'])
    df = df[(df['OddsH'] > 1) & (df['OddsA'] > 1)]

    # Calculate Overround (Vigorish)
    # Market margin sum of implied probs
    odds_d = df['OddsD'] if 'OddsD' in df.columns else np.inf
    imp_d = 1/odds_d
    imp_d = imp_d.replace(0, 0) # Handle inf
    
    df['Overround'] = (1/df['OddsH']) + (1/df['OddsA']) + imp_d
    
    mean_overround = df['Overround'].mean()
    
    print(f"Mean Overround: {mean_overround:.4f}")
    
    # Stationarity Test
    residuals = df['Overround'].values
    try:
        adf = adfuller(residuals)
        is_stationary = adf[1] < 0.05
    except:
        is_stationary = False
        adf = [0, 1]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Overround'].rolling(100).mean())
    plt.title('Rolling Overround (100 games)')
    plt.savefig('src/research/overround.png')
    
    readme = f"""# Odds Overround Analysis

## Goal
Measure the "tax" (vigorish) charged by bookmakers and check if it changes over time.

## Findings
- **Mean Overround**: {mean_overround:.4f} (Avg cost per bet)
- **Stationary?**: {'YES' if is_stationary else 'NO'} (p={adf[1]:.4e})

## Implication
A lower overround means easier to beat. If it trends down, the market is becoming more efficient/competitive.
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/OVERROUND_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_odds_overround()
