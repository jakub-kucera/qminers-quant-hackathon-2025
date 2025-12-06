import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

def analyze_roi_target():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    # Target: Return on Investment (Odds * Outcome)
    # If we bet 1 unit, what do we get back?
    # Home Win -> OddsH, Loss -> 0
    df['H_ROI'] = df['OddsH'] * df['H']
    df['A_ROI'] = df['OddsA'] * df['A'] # Assuming A=1 or A=0 logic is handled
    # If 'A' column missing, handle likely 'H' inverse
    if 'A' not in df.columns:
        df['A'] = 1 - df['H']
        df['A_ROI'] = df['OddsA'] * df['A']

    # Features: Simple Goal Diff Rolling
    df['DiffH'] = df['HS'] - df['AS'] if 'HS' in df.columns else df['HSC'] - df['ASC']
    df['DiffA'] = -df['DiffH']
    
    # Calculate Rolling Avg Diff per team
    # (Simplified for single script - real feature store would be better)
    # We will just do a simple non-leaking loop or group apply
    teams = pd.concat([df['HID'], df['AID']]).unique()
    team_stats = {t: [] for t in teams}
    
    features = []
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Iterate to avoid lookahead
    # This is slow but safe/correct
    for i, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        
        # Get history avg
        h_vals = team_stats[hid]
        a_vals = team_stats[aid]
        
        h_avg = np.mean(h_vals[-10:]) if h_vals else 0 # Last 10 games
        a_avg = np.mean(a_vals[-10:]) if a_vals else 0
        
        features.append([h_avg, a_avg])
        
        # Update
        team_stats[hid].append(row['DiffH'])
        team_stats[aid].append(row['DiffA'])
        
    X = np.array(features)
    y = df['H_ROI'].values # Predicting ROI directly
    
    # Train/Test Split (Time based)
    split = int(len(df) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    odds_h_test = df['OddsH'].iloc[split:].values
    outcomes_test = df['H'].iloc[split:].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds_roi = model.predict(X_test)
    
    # Strategy: Bet if Predicted ROI > 1.1 (10% edge)
    bets = preds_roi > 1.1
    
    n_bets = sum(bets)
    pnl = 0
    roi = 0
    
    if n_bets > 0:
        # PnL = Sum(Returns) - Cost
        returns = outcomes_test[bets] * odds_h_test[bets]
        cost = n_bets
        pnl = sum(returns) - cost
        roi = pnl / cost
        
    print(f"ROI Targeting Model Results:")
    print(f"Bets Placed: {n_bets} / {len(y_test)}")
    print(f"PnL: {pnl:.2f}")
    print(f"ROI: {roi:.2%}")
    
    readme = f"""# ROI Target Model (Direct Regression)

## Goal
Instead of predicting "Probability of Winning", we predict "Expected Return (ROI)" directly using Regression.
Features: Rolling Goal Differential (Last 10 games).

## Results
- **Bets**: {n_bets}
- **PnL**: {pnl:.2f} units
- **ROI**: {roi:.2%}

## Insight
Training directly on ROI (Odds * Outcome) allows the model to inherently value "Longshots" if they pay off enough.
"""
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/ROI_MODEL_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_roi_target()
