import pandas as pd
import numpy as np
import os

def analyze_momentum():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate Streak (Last 5 Games Wins)
    # 0 = Loss, 1 = Win
    # Need to normalize Home/Away outcomes to Team Win/Loss
    
    team_history = {} # TeamID -> List of [1,0,1...]
    
    h_streak = []
    a_streak = []
    
    for i, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        h_win = row['H']
        a_win = 1 - h_win # Approx
        
        # Get Streak Sum (Last 5)
        h_hist = team_history.get(hid, [])
        a_hist = team_history.get(aid, [])
        
        h_val = sum(h_hist[-5:]) if len(h_hist) >= 1 else 0
        a_val = sum(a_hist[-5:]) if len(a_hist) >= 1 else 0
        
        h_streak.append(h_val)
        a_streak.append(a_val)
        
        # Update
        if hid not in team_history: team_history[hid] = []
        if aid not in team_history: team_history[aid] = []
        
        team_history[hid].append(h_win)
        team_history[aid].append(a_win)
        
    df['StreakH'] = h_streak
    df['StreakA'] = a_streak
    
    # Analyze Impact
    # Group by Streak Value (0 to 5)
    
    # Performance vs Expectation? Or just raw win rate?
    # Raw win rate is biased by team strength (Good teams have streaks AND win more).
    # We should compare against Implied Probability (Expectation)
    
    df['Imp_H'] = 1/df['OddsH'] / (1/df['OddsH'] + 1/df['OddsA'])
    
    # Calculate Performance over Expectation (PoE)
    df['PoE'] = df['H'] - df['Imp_H']
    
    grouped = df.groupby('StreakH')['PoE'].mean()
    counts = df.groupby('StreakH')['PoE'].count()
    
    res_df = pd.DataFrame({'Streak': grouped.index, 'Avg_Overperf': grouped.values, 'Games': counts.values})
    
    print(res_df.to_string(index=False))
    
    readme = f"""# Momentum Analysis (Streaks)

## Goal
Does a winning streak (Last 5 games) provide "Momentum" that exceeds bookmaker expectations?

## Method
- Calculated sum of wins in last 5 games (Streak: 0-5).
- Compared Actual Result vs Implied Probability (Overperformance).

## Results
```
{res_df.to_string(index=False)}
```

## Implication
- **Positive Overperformance** for high streaks (4, 5) suggests Momentum is real and undervalued by the market.
- **Negative/Zero** suggests Gambler's Fallacy or Market Efficiency (priced in).
"""

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/MOMENTUM_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_momentum()
