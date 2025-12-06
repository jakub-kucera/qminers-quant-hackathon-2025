"""
Visualize Feature Weights for Simple Rolling Model

This script analyzes the feature weights log and creates visualizations showing:
- Feature importance over time
- Weight evolution for each feature
- Correlation between weights and bankroll
- Model accuracy over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the log file
log_file = "simple_rolling_weights_log.csv"
df = pd.read_csv(log_file)

print(f"Loaded {len(df)} log entries")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Bankroll range: ${df['bankroll'].min():.2f} to ${df['bankroll'].max():.2f}")
print(f"Final accuracy: {df['accuracy'].iloc[-1]:.2%}")

# Get feature columns (all columns starting with 'weight_')
feature_cols = [col for col in df.columns if col.startswith('weight_')]
feature_names = [col.replace('weight_', '') for col in feature_cols]

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Bankroll and Accuracy Over Time
ax1 = plt.subplot(3, 2, 1)
ax1_twin = ax1.twinx()
ax1.plot(df['game_num'], df['bankroll'], 'b-', linewidth=2, label='Bankroll')
ax1_twin.plot(df['game_num'], df['accuracy'], 'r-', linewidth=2, label='Accuracy')
ax1.set_xlabel('Game Number')
ax1.set_ylabel('Bankroll ($)', color='b')
ax1_twin.set_ylabel('Accuracy', color='r')
ax1.set_title('Bankroll and Accuracy Over Time')
ax1.grid(True, alpha=0.3)

# 2. Feature Importance (Final Weights)
ax2 = plt.subplot(3, 2, 2)
final_weights = df[feature_cols].iloc[-1].values
sorted_idx = np.argsort(np.abs(final_weights))[::-1]
top_features = [feature_names[i] for i in sorted_idx[:10]]
top_weights = [final_weights[i] for i in sorted_idx[:10]]

colors = ['green' if w > 0 else 'red' for w in top_weights]
ax2.barh(range(len(top_features)), top_weights, color=colors, alpha=0.7)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features)
ax2.set_xlabel('Weight')
ax2.set_title('Top 10 Most Important Features (Final Weights)')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3)

# 3. Weight Evolution for Top 5 Features
ax3 = plt.subplot(3, 2, 3)
top_5_idx = sorted_idx[:5]
for idx in top_5_idx:
    ax3.plot(df['game_num'], df[feature_cols[idx]], label=feature_names[idx], linewidth=2)
ax3.set_xlabel('Game Number')
ax3.set_ylabel('Weight')
ax3.set_title('Weight Evolution - Top 5 Features')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 4. Heatmap of Feature Weights Over Time (sampled)
ax4 = plt.subplot(3, 2, 4)
# Sample every 50 games to make heatmap readable
sample_interval = max(1, len(df) // 50)
sampled_df = df.iloc[::sample_interval]
weights_matrix = sampled_df[feature_cols].T.values

im = ax4.imshow(weights_matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax4.set_yticks(range(len(feature_names)))
ax4.set_yticklabels(feature_names, fontsize=8)
ax4.set_xlabel('Time (sampled)')
ax4.set_title('Feature Weights Heatmap Over Time')
plt.colorbar(im, ax=ax4, label='Weight')

# 5. Weight Stability (Standard Deviation)
ax5 = plt.subplot(3, 2, 5)
weight_std = df[feature_cols].std().values
sorted_std_idx = np.argsort(weight_std)[::-1]
top_volatile = [feature_names[i] for i in sorted_std_idx[:10]]
top_std = [weight_std[i] for i in sorted_std_idx[:10]]

ax5.barh(range(len(top_volatile)), top_std, color='orange', alpha=0.7)
ax5.set_yticks(range(len(top_volatile)))
ax5.set_yticklabels(top_volatile)
ax5.set_xlabel('Standard Deviation')
ax5.set_title('Most Volatile Features (Weight Std Dev)')
ax5.grid(True, alpha=0.3)

# 6. Correlation: Feature Weights vs Bankroll
ax6 = plt.subplot(3, 2, 6)
correlations = []
for col in feature_cols:
    corr = df[col].corr(df['bankroll'])
    correlations.append(corr)

sorted_corr_idx = np.argsort(np.abs(correlations))[::-1]
top_corr_features = [feature_names[i] for i in sorted_corr_idx[:10]]
top_corr = [correlations[i] for i in sorted_corr_idx[:10]]

colors_corr = ['green' if c > 0 else 'red' for c in top_corr]
ax6.barh(range(len(top_corr_features)), top_corr, color=colors_corr, alpha=0.7)
ax6.set_yticks(range(len(top_corr_features)))
ax6.set_yticklabels(top_corr_features)
ax6.set_xlabel('Correlation with Bankroll')
ax6.set_title('Features Most Correlated with Bankroll')
ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_rolling_feature_weights_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to: simple_rolling_feature_weights_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("FEATURE WEIGHT ANALYSIS")
print("="*60)

print("\nTop 10 Most Important Features (by absolute weight):")
for i, (feat, weight) in enumerate(zip(top_features, top_weights), 1):
    print(f"  {i:2d}. {feat:20s}: {weight:+.4f}")

print("\nTop 10 Most Volatile Features (by std dev):")
for i, (feat, std) in enumerate(zip(top_volatile, top_std), 1):
    print(f"  {i:2d}. {feat:20s}: {std:.4f}")

print("\nTop 10 Features Correlated with Bankroll:")
for i, (feat, corr) in enumerate(zip(top_corr_features, top_corr), 1):
    print(f"  {i:2d}. {feat:20s}: {corr:+.4f}")

print("\n" + "="*60)
print(f"Final Model Intercept: {df['intercept'].iloc[-1]:.4f}")
print(f"Final Accuracy: {df['accuracy'].iloc[-1]:.2%}")
print(f"Total Predictions: {df['predictions_made'].iloc[-1]:.0f}")
print(f"Final Bankroll: ${df['bankroll'].iloc[-1]:.2f}")
print(f"Profit: ${df['bankroll'].iloc[-1] - df['bankroll'].iloc[0]:.2f}")
print("="*60)
