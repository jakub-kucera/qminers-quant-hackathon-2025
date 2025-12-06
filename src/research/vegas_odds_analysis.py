import pandas as pd
import numpy as np
import os

def analyze_vegas_accuracy():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    # Ensure numeric columns
    for col in ['OddsH', 'OddsA', 'OddsD']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['OddsH', 'OddsA'])
    
    # Calculate Implied Probabilities (with vig removal)
    # Simple method: Proportional normalization
    df['Imp_H'] = 1/df['OddsH']
    df['Imp_A'] = 1/df['OddsA']
    df['Imp_D'] = 1/df['OddsD'] if 'OddsD' in df.columns else 0
    
    df['Total_Imp'] = df['Imp_H'] + df['Imp_A'] + df['Imp_D']
    
    df['Prob_H'] = df['Imp_H'] / df['Total_Imp']
    df['Prob_A'] = df['Imp_A'] / df['Total_Imp']
    
    # Actual Outcomes (1 = Home Win, 0 = Away Win/Draw)
    # Note: 'H' column is usually Home Win boolean
    df['Outcome_H'] = df['H'].astype(int)
    
    # Check for NaNs before MSE/Accuracy calculation
    df = df.dropna(subset=['Prob_H', 'Outcome_H'])

    # Calculate Brier Score (MSE)
    mse = ((df['Prob_H'] - df['Outcome_H'])**2).mean()
    
    # Calculate Log Loss
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    pred = df['Prob_H'].clip(epsilon, 1-epsilon)
    log_loss = -(df['Outcome_H'] * np.log(pred) + (1-df['Outcome_H']) * np.log(1-pred)).mean()
    
    # Accuracy (if we just picked the favorite)
    df['Pred_Correct'] = ((df['Prob_H'] > 0.5) == df['Outcome_H'])
    accuracy = df['Pred_Correct'].mean()
    
    print(f"Vegas Odds Analysis ({len(df)} games):")
    print(f"MSE (Brier Score): {mse:.5f}")
    print(f"Log Loss: {log_loss:.5f}")
    print(f"Accuracy (Favorite Wins): {accuracy:.2%}")
    
    # Expected Calibration
    # If Vegas says 60%, do they win 60%?
    # Ensure no NaN/Inf before casting
    df['Bucket'] = (df['Prob_H'] * 20).astype(int) / 20 # 5% buckets
    calibration = df.groupby('Bucket')['Outcome_H'].mean()
    counts = df.groupby('Bucket')['Outcome_H'].count()
    
    cal_df = pd.DataFrame({'Predicted': calibration.index + 0.025, 'Actual': calibration.values, 'Count': counts.values})
    
    readme = f"""# Vegas Odds Accuracy Analysis

## Goal
Determine how accurate the closing odds are. This sets a "baseline" for our models. It is very hard to beat Vegas.

## Results
- **MSE (Brier Score)**: {mse:.5f} (Lower is better)
- **Log Loss**: {log_loss:.5f}
- **Baseline Accuracy**: {accuracy:.2%} (If you just bet on the favorite)

## Calibration
How well do the odds match reality?
```
{cal_df.to_string(index=False)}
```

## Implication
Our simplified model needs to approach this MSE to be competitive. If our MSE is significantly higher, we are missing information that Vegas has.
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/VEGAS_ODDS_README.md', 'w') as f:
        f.write(readme)
    print("README generated at src/research/VEGAS_ODDS_README.md")

if __name__ == "__main__":
    analyze_vegas_accuracy()
