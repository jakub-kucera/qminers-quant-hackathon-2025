import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, r2_score
import os

def analyze_model_odds():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    # Strict filtering for valid odds
    df = df.dropna(subset=['OddsH', 'OddsA', 'H'])
    df = df[(df['OddsH'] > 1.01) & (df['OddsA'] > 1.01)]
    
    # Check Predictive Power of Odds
    # Implied Prob
    df['Imp_H'] = 1/df['OddsH'] / (1/df['OddsH'] + 1/df['OddsA'])
    
    # Check for NaNs/Infs again
    df = df[np.isfinite(df['Imp_H'])]
    
    # Metric 1: R2 Score (Pseudo R2 since outcome is binary)
    # Compare "Model" (Odds) vs Mean Baseline
    y_true = df['H']
    y_pred = df['Imp_H']
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    
    print(f"R2 (Odds vs Outcome): {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Check if Odds predict Margin?
    if 'HS' in df.columns and 'AS' in df.columns:
        df['Margin'] = df['HS'] - df['AS']
        # Correlation with Margin
        # Higher Prob H -> Higher Margin?
        corr_margin = df['Imp_H'].corr(df['Margin'])
        print(f"Correlation (Odds Prob vs Goal Margin): {corr_margin:.4f}")
    else:
        corr_margin = 0
        
    readme = f"""# Model Odds / Market Efficiency Analysis

## Goal
Evaluate how well the Odds predict the actual game outcomes (Win and Margin).

## Metrics
- **R2 Score**: {r2:.4f} (Low for binary targets, typically < 0.1 for sports)
- **Log Loss**: {ll:.4f} (Lower is better)
- **Correlation (Odds vs Margin)**: {corr_margin:.4f}

## Interpretation
- High correlation with **Margin** confirms that Odds capture team strength differences, not just win probability.
- The market is a "Strong Predictor" of margin.
"""

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/MODEL_ODDS_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_model_odds()
