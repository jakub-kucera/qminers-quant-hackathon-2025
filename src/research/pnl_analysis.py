import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_pnl():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    # Simulate PnL of "Betting on Home Favorites" strategy
    # Simple strategy: If OddsH < 2.0 (Implied > 50%), bet 1 unit.
    
    df['Bet'] = (df['OddsH'] < 1.8) # Conservative favorite
    
    pnl = []
    curr_pnl = 0
    
    for i, row in df.iterrows():
        if row['Bet']:
            curr_pnl -= 1 # Cost
            if row['H']:
                curr_pnl += row['OddsH']
        pnl.append(curr_pnl)
        
    df['PnL'] = pnl
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['PnL'])
    plt.title('PnL of Betting Home Favorites (< 1.8)')
    plt.xlabel('Games')
    plt.ylabel('Profit (Units)')
    plt.grid()
    plt.savefig('src/research/pnl_history.png')
    
    final_pnl = df['PnL'].iloc[-1]
    n_bets = df['Bet'].sum()
    roi = final_pnl / n_bets if n_bets > 0 else 0
    
    readme = f"""# PnL Analysis (Baseline Strategy)

## Strategy
Bet 1 Unit on Home Team if Odds < 1.8 (Strong Favorite).

## Results
- **Total Bets**: {n_bets}
- **Final PnL**: {final_pnl:.2f} Units
- **ROI**: {roi:.2%}

## Insight
- If PnL trends downward (negative ROI), simply backing favorites is a losing strategy (due to Vig).
- See `pnl_history.png` for equity curve.
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/PNL_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_pnl()
