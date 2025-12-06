"""
Diagnostic script to check cached predictions
"""

import pickle
import pandas as pd

# Load cache
print("Loading cache...")
with open('model_predictions_cache.pkl', 'rb') as f:
    cached_data = pickle.load(f)

predictions = cached_data['predictions']
games = cached_data['games']

print(f"\n📊 Cache Statistics:")
print(f"  Total games: {len(games)}")
print(f"  Total predictions: {len(predictions)}")
print(f"  Prediction rate: {len(predictions)/len(games)*100:.1f}%")

# Sample some predictions
print(f"\n🎯 Sample Predictions:")
for i, (idx, prob) in enumerate(list(predictions.items())[:10]):
    game = games.loc[idx]
    print(f"  Game {idx}: {prob:.3f} (Odds: {game['OddsH']:.2f}, Actual: {'Home' if game['H'] else 'Away'})")

# Check prediction distribution
import numpy as np
probs = list(predictions.values())
print(f"\n📈 Prediction Distribution:")
print(f"  Min: {min(probs):.3f}")
print(f"  Max: {max(probs):.3f}")
print(f"  Mean: {np.mean(probs):.3f}")
print(f"  Median: {np.median(probs):.3f}")

# Count how many would bet with different thresholds
print(f"\n🎲 Betting Opportunities (with different thresholds):")
for min_prob in [0.5, 0.6, 0.7, 0.8, 0.9]:
    count = sum(1 for p in probs if p > min_prob)
    print(f"  min_probability={min_prob}: {count} bets ({count/len(probs)*100:.1f}%)")

# Check edges
print(f"\n💰 Edge Analysis:")
edges = []
for idx, prob in predictions.items():
    game = games.loc[idx]
    bookmaker_prob = 1.0 / game['OddsH']
    edge = prob - bookmaker_prob
    edges.append(edge)

print(f"  Positive edges: {sum(1 for e in edges if e > 0)} ({sum(1 for e in edges if e > 0)/len(edges)*100:.1f}%)")
print(f"  Edges > 2%: {sum(1 for e in edges if e > 0.02)} ({sum(1 for e in edges if e > 0.02)/len(edges)*100:.1f}%)")
print(f"  Edges > 5%: {sum(1 for e in edges if e > 0.05)} ({sum(1 for e in edges if e > 0.05)/len(edges)*100:.1f}%)")
print(f"  Edges > 10%: {sum(1 for e in edges if e > 0.10)} ({sum(1 for e in edges if e > 0.10)/len(edges)*100:.1f}%)")

# Find best opportunities
print(f"\n🌟 Best Betting Opportunities:")
opportunities = []
for idx, prob in predictions.items():
    game = games.loc[idx]
    bookmaker_prob = 1.0 / game['OddsH']
    edge = prob - bookmaker_prob
    if edge > 0.05 and prob > 0.6:
        opportunities.append((idx, prob, edge, game['OddsH'], game['H']))

opportunities.sort(key=lambda x: x[2], reverse=True)
for idx, prob, edge, odds, won in opportunities[:10]:
    result = "✅ Won" if won else "❌ Lost"
    print(f"  Game {idx}: Prob={prob:.3f}, Edge={edge:.3f}, Odds={odds:.2f} {result}")
