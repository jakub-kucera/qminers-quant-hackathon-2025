import pandas as pd

# from model_pepe import Model
# from model_hybrid import Model

# from model import Model
# from model_bookie_rookie import Model
# from model_simple_rolling import Model
from model_hybrid import Model
from environment import Environment

games = pd.read_csv("../data/train_games.csv", index_col=0, parse_dates=["Date", "Open"])
# games = pd.read_csv("../data/bookie_rookie_games.csv", index_col=0, parse_dates=["Date", "Open"])

model = Model()
env = Environment(games, model, init_bankroll=1000, min_bet=1, max_bet=250)

evaluation = env.run()

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
    
    if model.accuracy_history:
        print(f"\nAccuracy Over Time (every 50 predictions):")
        for i, acc in enumerate(model.accuracy_history):
            print(f"  After {(i+1)*50} predictions: {acc:.2%}")

print("=" * 60)

history = env.get_history()
