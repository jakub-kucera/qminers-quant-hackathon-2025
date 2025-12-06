"""
Analyze Data Changes Around Game 2400

This script investigates what changed in the data around game 2400 (±200 games).
It looks at feature weights, model performance, and data characteristics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Configuration
CENTER_GAME = 2400
WINDOW = 200
START_GAME = CENTER_GAME - WINDOW
END_GAME = CENTER_GAME + WINDOW

print(f"Analyzing games {START_GAME} to {END_GAME} (centered on {CENTER_GAME})")

# Load the weight log
print("\nLoading weight log...")
df = pd.read_csv("simple_rolling_weights_log.csv")

# Filter to the window
df_window = df[(df['game_num'] >= START_GAME) & (df['game_num'] <= END_GAME)].copy()

if len(df_window) == 0:
    print(f"❌ No data found in range {START_GAME}-{END_GAME}")
    print(f"Available range: {df['game_num'].min()} to {df['game_num'].max()}")
    exit(1)

print(f"Found {len(df_window)} log entries in the window")

# Get feature columns
feature_cols = [col for col in df.columns if col.startswith('weight_')]
feature_names = [col.replace('weight_', '') for col in feature_cols]

# Create visualizations
fig = plt.figure(figsize=(24, 16))

# 1. Bankroll Around Game 2400
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df_window['game_num'], df_window['bankroll'], linewidth=2, color='blue')
ax1.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax1.set_xlabel('Game Number')
ax1.set_ylabel('Bankroll ($)')
ax1.set_title(f'Bankroll Around Game {CENTER_GAME}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Accuracy Around Game 2400
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df_window['game_num'], df_window['accuracy'], linewidth=2, color='green')
ax2.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax2.set_xlabel('Game Number')
ax2.set_ylabel('Accuracy')
ax2.set_title(f'Model Accuracy Around Game {CENTER_GAME}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Predictions Made
ax3 = plt.subplot(3, 3, 3)
ax3.plot(df_window['game_num'], df_window['predictions_made'], linewidth=2, color='purple')
ax3.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax3.set_xlabel('Game Number')
ax3.set_ylabel('Total Predictions')
ax3.set_title(f'Cumulative Predictions Around Game {CENTER_GAME}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Weight Changes - Top 5 Features
ax4 = plt.subplot(3, 3, 4)
# Find features with most change in this window
weight_changes = {}
for col in feature_cols:
    change = abs(df_window[col].iloc[-1] - df_window[col].iloc[0])
    weight_changes[col] = change

top_5_changing = sorted(weight_changes.items(), key=lambda x: x[1], reverse=True)[:5]
for col, _ in top_5_changing:
    feature_name = col.replace('weight_', '')
    ax4.plot(df_window['game_num'], df_window[col], label=feature_name, linewidth=2)

ax4.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Game Number')
ax4.set_ylabel('Weight')
ax4.set_title('Top 5 Most Changing Features')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Intercept Evolution
ax5 = plt.subplot(3, 3, 5)
ax5.plot(df_window['game_num'], df_window['intercept'], linewidth=2, color='orange')
ax5.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax5.set_xlabel('Game Number')
ax5.set_ylabel('Intercept')
ax5.set_title(f'Model Intercept Around Game {CENTER_GAME}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Weight Volatility (Rolling Std)
ax6 = plt.subplot(3, 3, 6)
# Calculate rolling std of all weights
window_size = 20
weight_volatility = df_window[feature_cols].rolling(window=window_size).std().mean(axis=1)
ax6.plot(df_window['game_num'].iloc[window_size-1:], weight_volatility.iloc[window_size-1:], 
         linewidth=2, color='red')
ax6.axvline(CENTER_GAME, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax6.set_xlabel('Game Number')
ax6.set_ylabel('Average Weight Std Dev')
ax6.set_title(f'Weight Volatility (Rolling {window_size} games)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Heatmap of Weight Changes
ax7 = plt.subplot(3, 3, 7)
# Sample every 10 games for readability
sample_interval = max(1, len(df_window) // 40)
sampled = df_window.iloc[::sample_interval]
weights_matrix = sampled[feature_cols].T.values

im = ax7.imshow(weights_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax7.set_yticks(range(len(feature_names)))
ax7.set_yticklabels(feature_names, fontsize=7)
ax7.set_xlabel('Time (sampled)')
ax7.set_title(f'Feature Weights Heatmap (Games {START_GAME}-{END_GAME})')
plt.colorbar(im, ax=ax7, label='Weight')

# Mark center
center_idx = len(sampled) // 2
ax7.axvline(center_idx, color='yellow', linestyle='--', linewidth=2, alpha=0.7)

# 8. Bankroll Change Rate
ax8 = plt.subplot(3, 3, 8)
bankroll_change = df_window['bankroll'].diff()
ax8.plot(df_window['game_num'].iloc[1:], bankroll_change.iloc[1:], linewidth=1, color='blue', alpha=0.7)
ax8.axhline(0, color='black', linestyle='-', linewidth=1)
ax8.axvline(CENTER_GAME, color='red', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax8.set_xlabel('Game Number')
ax8.set_ylabel('Bankroll Change ($)')
ax8.set_title('Bankroll Change per Game')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Feature Importance Shift
ax9 = plt.subplot(3, 3, 9)
# Compare weights before and after center
before_idx = df_window['game_num'] < CENTER_GAME
after_idx = df_window['game_num'] >= CENTER_GAME

if before_idx.sum() > 0 and after_idx.sum() > 0:
    weights_before = df_window.loc[before_idx, feature_cols].iloc[-1].values
    weights_after = df_window.loc[after_idx, feature_cols].iloc[0].values
    weight_diff = weights_after - weights_before
    
    sorted_idx = np.argsort(np.abs(weight_diff))[::-1][:10]
    top_features = [feature_names[i] for i in sorted_idx]
    top_diffs = [weight_diff[i] for i in sorted_idx]
    
    colors = ['green' if d > 0 else 'red' for d in top_diffs]
    ax9.barh(range(len(top_features)), top_diffs, color=colors, alpha=0.7)
    ax9.set_yticks(range(len(top_features)))
    ax9.set_yticklabels(top_features, fontsize=8)
    ax9.set_xlabel('Weight Change')
    ax9.set_title(f'Biggest Weight Changes at Game {CENTER_GAME}')
    ax9.axvline(0, color='black', linestyle='--', linewidth=1)
    ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'game_{CENTER_GAME}_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: game_{CENTER_GAME}_analysis.png")

# Print detailed statistics
print("\n" + "="*80)
print(f"ANALYSIS AROUND GAME {CENTER_GAME}")
print("="*80)

# Before and after stats
before_data = df_window[df_window['game_num'] < CENTER_GAME]
after_data = df_window[df_window['game_num'] >= CENTER_GAME]

if len(before_data) > 0 and len(after_data) > 0:
    print(f"\n📊 BEFORE Game {CENTER_GAME} (last 10 games):")
    before_recent = before_data.tail(10)
    print(f"  Average bankroll: ${before_recent['bankroll'].mean():.2f}")
    print(f"  Average accuracy: {before_recent['accuracy'].mean():.2%}")
    print(f"  Bankroll trend: ${before_recent['bankroll'].iloc[-1] - before_recent['bankroll'].iloc[0]:.2f}")
    
    print(f"\n📊 AFTER Game {CENTER_GAME} (first 10 games):")
    after_recent = after_data.head(10)
    print(f"  Average bankroll: ${after_recent['bankroll'].mean():.2f}")
    print(f"  Average accuracy: {after_recent['accuracy'].mean():.2%}")
    print(f"  Bankroll trend: ${after_recent['bankroll'].iloc[-1] - after_recent['bankroll'].iloc[0]:.2f}")
    
    print(f"\n📈 CHANGE:")
    bankroll_change = after_recent['bankroll'].mean() - before_recent['bankroll'].mean()
    accuracy_change = after_recent['accuracy'].mean() - before_recent['accuracy'].mean()
    print(f"  Bankroll: ${bankroll_change:+.2f}")
    print(f"  Accuracy: {accuracy_change:+.2%}")

# Top weight changes
print(f"\n🔄 TOP 10 WEIGHT CHANGES at Game {CENTER_GAME}:")
if before_idx.sum() > 0 and after_idx.sum() > 0:
    for i, (feat, diff) in enumerate(zip(top_features, top_diffs), 1):
        direction = "↑" if diff > 0 else "↓"
        print(f"  {i:2d}. {feat:20s}: {diff:+.4f} {direction}")

# Check for anomalies
print(f"\n⚠️  ANOMALY DETECTION:")

# Large bankroll changes
large_changes = df_window[abs(df_window['bankroll'].diff()) > 100]
if len(large_changes) > 0:
    print(f"  Found {len(large_changes)} games with bankroll change > $100:")
    for idx, row in large_changes.iterrows():
        change = row['bankroll'] - df_window.loc[idx-1, 'bankroll'] if idx > 0 else 0
        print(f"    Game {row['game_num']}: {change:+.2f}")

# Accuracy drops
accuracy_drops = df_window[df_window['accuracy'].diff() < -0.05]
if len(accuracy_drops) > 0:
    print(f"  Found {len(accuracy_drops)} games with accuracy drop > 5%:")
    for idx, row in accuracy_drops.iterrows():
        drop = row['accuracy'] - df_window.loc[idx-1, 'accuracy'] if idx > 0 else 0
        print(f"    Game {row['game_num']}: {drop:+.2%}")

# Load actual games data to see what happened
print(f"\n🎮 ACTUAL GAMES DATA:")
try:
    games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date"])
    
    # Find games around this index
    game_window = games.iloc[START_GAME:END_GAME+1]
    
    print(f"  Date range: {game_window['Date'].min().date()} to {game_window['Date'].max().date()}")
    print(f"  Seasons: {game_window['Season'].unique()}")
    print(f"  Total goals avg: {(game_window['HS'] + game_window['AS']).mean():.2f}")
    print(f"  Home win rate: {game_window['H'].mean():.2%}")
    
    # Check for season boundary
    if game_window['Season'].nunique() > 1:
        print(f"\n  ⚠️  SEASON CHANGE DETECTED!")
        for season in sorted(game_window['Season'].unique()):
            season_games = game_window[game_window['Season'] == season]
            print(f"    Season {season}: {len(season_games)} games")
    
except Exception as e:
    print(f"  Could not load games data: {e}")

print("\n" + "="*80)
print(f"Analysis complete! Check game_{CENTER_GAME}_analysis.png for visualizations")
print("="*80)
