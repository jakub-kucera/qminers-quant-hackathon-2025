import pandas as pd
from model_hybrid import Model
from environment import Environment

games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

env = Environment(games, Model(), init_bankroll=1000, min_bet=5, max_bet=100)

# Run and track when bets are placed
print("Running evaluation...")
evaluation = env.run()

print()
print(f"Final bankroll: {env.bankroll:.2f}")

history = env.get_history()

# Analyze when bets were placed
bet_dates = []
for idx, row in history.iterrows():
    if row['BetH'] > 0 or row['BetA'] > 0 or row['BetD'] > 0:
        bet_dates.append(row['Date'])

if bet_dates:
    print(f"\nTotal bets placed: {len(bet_dates)}")
    print(f"First bet: {min(bet_dates)}")
    print(f"Last bet: {max(bet_dates)}")
    
    # Group by year
    bet_df = pd.DataFrame({'Date': bet_dates})
    bet_df['Year'] = bet_df['Date'].dt.year
    print("\nBets by year:")
    print(bet_df['Year'].value_counts().sort_index())
else:
    print("\nNo bets were placed!")
