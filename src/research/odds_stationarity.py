import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

def analyze_odds_stationarity():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
    
    # Filter valid lines
    df = df.dropna(subset=['OddsH', 'OddsA', 'H'])
    df = df[(df['OddsH'] > 1.01) & (df['OddsA'] > 1.01)] # Basic sanity check
    
    # Columns
    df['Imp_H'] = 1/df['OddsH']
    df['Imp_A'] = 1/df['OddsA']
    df['Total_Imp'] = df['Imp_H'] + df['Imp_A']
    
    df['True_H'] = df['Imp_H'] / df['Total_Imp']
    
    # Calculate Residuals (Prediction Error)
    # MSE of market prediction
    # If market is perfectly efficient, residuals should be white noise
    df['Error'] = (df['True_H'] - df['H']) # Signed error
    
    residuals = df['Error'].values
    
    # ADF Test
    print("Running ADF Test...")
    try:
        adf_result = adfuller(residuals)
        stat = adf_result[0]
        p_val = adf_result[1]
        
        is_stationary = p_val < 0.05
    except Exception as e:
        print(f"ADF Failed: {e}")
        stat, p_val, is_stationary = 0, 1, False

    readme = f"""# Odds Stationarity Analysis (ADF Test)

## Goal
Check if the market efficiency (or error) is "stationary" over time.
Stationarity means the statistical properties (mean error, variance) do not change over time.

## Results
- **ADF Statistic**: {stat:.4f}
- **p-value**: {p_val:.4e}
- **Stationary?**: {'YES' if is_stationary else 'NO'}

## Interpretation
Since p-value < 0.05, we **{'Reject' if is_stationary else 'Fail to Reject'}** the Null Hypothesis.
- **Stationary (Yes)**: The market's error rate is stable. We can model it with a time-invariant model (like simple rolling averages).
- **Non-Stationary (No)**: The market is evolving (e.g., becoming more efficient). We might need models that adapt quickly (like recent-window only).
"""

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/ODDS_STATIONARITY.md', 'w') as f:
        f.write(readme)
    print("README generated at src/research/ODDS_STATIONARITY.md")

if __name__ == "__main__":
    analyze_odds_stationarity()
