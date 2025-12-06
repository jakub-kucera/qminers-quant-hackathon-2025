import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_odds_metrics():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    df = df.dropna(subset=['OddsH', 'OddsA', 'H'])
    
    # Implied Probs
    df['Imp_H'] = 1/df['OddsH']
    df['Imp_A'] = 1/df['OddsA']
    df['Vig'] = df['Imp_H'] + df['Imp_A'] - 1
    
    # Rolling Metrics
    window = 500
    df['Rolling_Vig'] = df['Vig'].rolling(window=window).mean()
    
    # Accuracy of Favorites
    df['Favorite_Won'] = ((df['Imp_H'] > df['Imp_A']) == (df['H'] == 1))
    df['Rolling_Acc'] = df['Favorite_Won'].rolling(window=window).mean()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Rolling_Vig'], label='Rolling Vig (Margin)')
    plt.plot(df['Rolling_Acc'], label='Rolling Favorite Accuracy')
    plt.legend()
    plt.title(f'Market Metrics (Rolling {window} games)')
    plt.savefig('src/research/market_metrics.png')
    
    avg_vig = df['Vig'].mean()
    
    readme = f"""# Odds Market Metrics

## Goal
Analyze the evolution of Bookmaker Margin (Vig) and Accuracy over time.

## Findings
- **Average Vigorish**: {avg_vig:.2%}
- **Evolution**: See `market_metrics.png`.

## Interpretation
- If Vig is decreasing, the market is getting tighter/more competitive.
- Accuracy fluctuating indicates seasons where favorites dominated (predictable) vs seasons of upsets (high variance).
"""
    
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/ODDS_METRICS.md', 'w') as f:
        f.write(readme)
    print("README generated at src/research/ODDS_METRICS.md")

if __name__ == "__main__":
    analyze_odds_metrics()
