"""
Example: Run Underdog Sniper Model

This script runs the Underdog Sniper model on training data.
The model targets high-EV underdogs (Odds >= 3.0, EV >= 20%).
"""

import pandas as pd
from model_underdog_sniper import Model
from environment import Environment

# Load data
print("Loading data...")
try:
    games = pd.read_csv("data/train_games.csv", index_col=0, parse_dates=["Date", "Open"])
except FileNotFoundError:
    games = pd.read_csv("../data/train_games.csv", index_col=0, parse_dates=["Date", "Open"])

# Create model
# Parameters from the prompt description
print("Initializing Underdog Sniper Model...")
model = Model(
    fixed_bet_size=250.0,
    ev_threshold=0.20,
    min_odds=3.0,
    min_calibration_samples=200
)

# Create environment
env = Environment(
    games=games,
    model=model,
    init_bankroll=1000.0,
    min_bet=1,
    max_bet=500  # Allow larger bets up to 500
)

# Run evaluation
print("Running Underdog Sniper Simulation...")
print(f"Start: {env.start_date}, End: {env.end_date}")
print("-" * 50)

env.run()

# Print results
print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Final Bankroll: ${env.bankroll:.2f}")
print(f"Profit: ${env.bankroll - 1000:.2f}")
print(f"ROI: {((env.bankroll - 1000) / 1000 * 100):.2f}%")

# Analyze bets
history = env.get_history()
print("\nHistory:")
print(history.tail())
