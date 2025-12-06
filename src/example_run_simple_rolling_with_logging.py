"""
Example: Run Simple Rolling Model with Feature Weight Logging

This script demonstrates how to run the simple rolling model with logging enabled.
It will create a CSV log file tracking feature weights and bankroll over time.
"""

import pandas as pd
from model_simple_rolling_with_logging import Model
from environment import Environment

# Load data
games = pd.read_csv("../data/round2_games.csv", index_col=0, parse_dates=["Date", "Open"])

# Create model with logging enabled
model = Model(
    log_file="simple_rolling_weights_log.csv",  # Where to save the log
    # Model parameters (use optimized values)
    # rolling_window_short=5,
    # rolling_window_med=10,
    # rolling_window_long=20,
    # required_games=100,
    # retrain_freq=50,
    # # Betting parameters
    # min_edge=0.1374,
    # max_kelly_fraction=0.0980,
    # kelly_multiplier=0.7867,
    # normalize_odds=True,
    # min_probability=0.8
)

# Create environment
env = Environment(
    games=games,
    model=model,
    init_bankroll=1000,
    min_bet=1,
    max_bet=250
)

# Run evaluation
print("Running evaluation with feature weight logging...")
print("This will create: simple_rolling_weights_log.csv")
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

# Print model accuracy
if hasattr(model, 'predictions_made') and model.predictions_made > 0:
    accuracy = model.predictions_correct / model.predictions_made
    print()
    print("=" * 60)
    print("MODEL ACCURACY")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Correct Predictions: {model.predictions_correct}/{model.predictions_made}")

print()
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. View the log file:")
print("   cat simple_rolling_weights_log.csv | head -20")
print()
print("2. Visualize feature weights:")
print("   python visualize_simple_rolling_weights.py")
print()
print("3. Analyze which features matter most!")
print("=" * 60)
