"""
Example: Run evaluation with ensemble weight logging

This script shows how to use the logging model to track ensemble weights.
"""

import pandas as pd
from model_hybrid_with_logging import Model
from environment import Environment

# Load games data
games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

# Create model with logging enabled
model = Model(
    log_file="ensemble_weights_log.csv",  # Where to save the log
    # ... all other parameters use defaults from model_hybrid.py
)

# Initialize environment
env = Environment(
    games=games.copy(),
    model=model,
    start_date=None,  # Start from beginning
    end_date=None,     # Run until end
    init_bankroll=1000.0,
    min_bet=5,
    max_bet=100
)

# Run evaluation
print("Running evaluation with ensemble weight logging...")
evaluation = env.run()

# Print results
print(f"\nFinal Bankroll: ${env.bankroll:.2f}")
print(f"Profit: ${env.bankroll - 1000:.2f}")
print(f"ROI: {((env.bankroll - 1000) / 1000 * 100):.2f}%")
print(f"\nEnsemble weights logged to: ensemble_weights_log.csv")
print("\nTo visualize the weights, run:")
print("  python visualize_ensemble_weights.py")
