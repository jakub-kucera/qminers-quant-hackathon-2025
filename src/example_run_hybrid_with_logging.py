"""
Example: Run Hybrid Model with Logging

This script runs the Hybrid model with logging enabled.
It will create 'hybrid_model_log.csv' tracking bankroll and ensemble weights over time.
Using train_games.csv as data source.
"""

import pandas as pd
import numpy as np
from model_hybrid_with_logging import Model
from environment import Environment

# Load data
print("Loading data...")
try:
    games = pd.read_csv("data/round1_games.csv", index_col=0, parse_dates=["Date", "Open"])
except FileNotFoundError:
    # Fallback if running from src directory
    games = pd.read_csv("../data/round1_games.csv", index_col=0, parse_dates=["Date", "Open"])

# Create model with logging enabled
# Note: Parameters are already set to defaults in model_hybrid.py (which are the requested Trial 4 values)
print("Initializing Hybrid Model (Trial 4 Params)...")
model = Model(
    log_file="hybrid_model_log.csv",
    # You can override params here if needed, but defaults are now updated
    start_season=2015 # Ensure we start betting from 2010
)

# Create environment
env = Environment(
    games=games,
    model=model,
    init_bankroll=1000.0,
    min_bet=1,
    max_bet=250
)

# Run evaluation
print("Running evaluation with logging...")
print("This will create: hybrid_model_log.csv")
print(f"Simulation Start Date: {env.start_date}")
print(f"Simulation End Date: {env.end_date}")
print()

env.run()

# Print results
print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Final Bankroll: ${env.bankroll:.2f}")
print(f"Profit: ${env.bankroll - 1000:.2f}")
print(f"ROI: {((env.bankroll - 1000) / 1000 * 100):.2f}%")

print()
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. View the log file:")
print("   cat hybrid_model_log.csv | head -20")
print()
print("2. Plot the results (Bankroll over time):")
print("   (You may want to create a visualization script for this)")
