"""
NaN Dashboard by Season

Creates a comprehensive dashboard with one chart per variable showing
NaN percentage across seasons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Load data
print("Loading games.csv...")
games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

print(f"Loaded {len(games)} games")
print(f"Seasons: {games['Season'].min():.0f} to {games['Season'].max():.0f}")

# Get all columns except index and dates
columns_to_analyze = [col for col in games.columns if col not in ['Date', 'Open']]

# Calculate NaN percentage by season for each column
seasons = sorted(games['Season'].unique())
nan_data = {}

for col in columns_to_analyze:
    nan_pcts = []
    for season in seasons:
        season_games = games[games['Season'] == season]
        nan_pct = season_games[col].isna().sum() / len(season_games) * 100
        nan_pcts.append(nan_pct)
    nan_data[col] = nan_pcts

# Filter to only columns that have at least some NaN values
columns_with_nan = [col for col in columns_to_analyze if max(nan_data[col]) > 0]

print(f"\nFound {len(columns_with_nan)} columns with NaN values")
print(f"Creating dashboard with {len(columns_with_nan)} charts...")

# Calculate grid size
n_cols = 4
n_rows = int(np.ceil(len(columns_with_nan) / n_cols))

# Create figure
fig = plt.figure(figsize=(24, 6 * n_rows))

# Create a chart for each column
for idx, col in enumerate(columns_with_nan, 1):
    ax = plt.subplot(n_rows, n_cols, idx)
    
    # Plot NaN percentage by season
    nan_pcts = nan_data[col]
    colors = ['red' if pct > 50 else 'orange' if pct > 10 else 'yellow' if pct > 0 else 'green' 
              for pct in nan_pcts]
    
    bars = ax.bar(range(len(seasons)), nan_pcts, color=colors, alpha=0.7, edgecolor='black')
    
    # Formatting
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([int(s) for s in seasons], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Season', fontsize=9)
    ax.set_ylabel('NaN %', fontsize=9)
    ax.set_title(f'{col}', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars if > 0
    for i, (bar, pct) in enumerate(zip(bars, nan_pcts)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, fc='green', alpha=0.7, label='0% NaN'),
    plt.Rectangle((0,0),1,1, fc='yellow', alpha=0.7, label='0-10% NaN'),
    plt.Rectangle((0,0),1,1, fc='orange', alpha=0.7, label='10-50% NaN'),
    plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, label='>50% NaN')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.suptitle('NaN Percentage by Season - All Variables', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('nan_dashboard_by_season.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved dashboard to: nan_dashboard_by_season.png")

# Print summary statistics
print("\n" + "="*80)
print("NaN SUMMARY BY VARIABLE")
print("="*80)

for col in columns_with_nan:
    nan_pcts = nan_data[col]
    avg_nan = np.mean(nan_pcts)
    max_nan = max(nan_pcts)
    seasons_with_nan = sum(1 for pct in nan_pcts if pct > 0)
    
    print(f"\n{col}:")
    print(f"  Average NaN across seasons: {avg_nan:.1f}%")
    print(f"  Maximum NaN in any season: {max_nan:.1f}%")
    print(f"  Seasons with NaN: {seasons_with_nan}/{len(seasons)}")
    
    # Show worst seasons
    if max_nan > 0:
        worst_seasons = [(seasons[i], nan_pcts[i]) for i in range(len(seasons)) if nan_pcts[i] > 10]
        if worst_seasons:
            print(f"  Seasons with >10% NaN:", end=" ")
            print(", ".join([f"{int(s)} ({p:.1f}%)" for s, p in worst_seasons]))

print("\n" + "="*80)
print("Dashboard complete!")
print("="*80)
