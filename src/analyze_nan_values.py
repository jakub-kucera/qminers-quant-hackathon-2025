"""
NaN Analysis for games.csv

This script analyzes all columns in games.csv to identify:
- Which columns have NaN values
- How many NaN values per column
- When NaN values occur (by season, date range)
- Patterns in missing data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Load data
print("Loading games.csv...")
games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

print(f"Loaded {len(games)} games")
print(f"Total columns: {len(games.columns)}")
print(f"Date range: {games['Date'].min().date()} to {games['Date'].max().date()}")

# Analyze NaN values
print("\n" + "="*80)
print("NaN ANALYSIS")
print("="*80)

# Count NaN values per column
nan_counts = games.isna().sum()
nan_percentages = (nan_counts / len(games) * 100).round(2)

# Create summary DataFrame
nan_summary = pd.DataFrame({
    'Column': nan_counts.index,
    'NaN_Count': nan_counts.values,
    'NaN_Percentage': nan_percentages.values,
    'Non_NaN_Count': len(games) - nan_counts.values
}).sort_values('NaN_Count', ascending=False)

# Print summary
print(f"\n📊 OVERALL SUMMARY:")
print(f"  Total columns: {len(games.columns)}")
print(f"  Columns with NaN: {(nan_counts > 0).sum()}")
print(f"  Columns without NaN: {(nan_counts == 0).sum()}")

print(f"\n⚠️  COLUMNS WITH NaN VALUES:")
columns_with_nan = nan_summary[nan_summary['NaN_Count'] > 0]
if len(columns_with_nan) > 0:
    for idx, row in columns_with_nan.iterrows():
        print(f"  {row['Column']:15s}: {row['NaN_Count']:6.0f} NaN ({row['NaN_Percentage']:6.2f}%) | {row['Non_NaN_Count']:6.0f} valid")
else:
    print("  No columns with NaN values!")

print(f"\n✅ COLUMNS WITHOUT NaN VALUES:")
columns_without_nan = nan_summary[nan_summary['NaN_Count'] == 0]
print(f"  {len(columns_without_nan)} columns: {', '.join(columns_without_nan['Column'].tolist())}")

# Analyze NaN patterns by season
print(f"\n📅 NaN VALUES BY SEASON:")
for season in sorted(games['Season'].unique()):
    season_games = games[games['Season'] == season]
    season_nan_counts = season_games.isna().sum()
    columns_with_nan_in_season = season_nan_counts[season_nan_counts > 0]
    
    if len(columns_with_nan_in_season) > 0:
        print(f"\n  Season {int(season)} ({len(season_games)} games):")
        for col, count in columns_with_nan_in_season.items():
            pct = count / len(season_games) * 100
            print(f"    {col:15s}: {count:4.0f} NaN ({pct:5.1f}%)")

# Create visualizations
fig = plt.figure(figsize=(24, 14))

# 1. NaN Count by Column
ax1 = plt.subplot(3, 2, 1)
columns_with_nan_sorted = columns_with_nan.sort_values('NaN_Count', ascending=True)
if len(columns_with_nan_sorted) > 0:
    ax1.barh(range(len(columns_with_nan_sorted)), columns_with_nan_sorted['NaN_Count'], 
             color='red', alpha=0.7)
    ax1.set_yticks(range(len(columns_with_nan_sorted)))
    ax1.set_yticklabels(columns_with_nan_sorted['Column'], fontsize=9)
    ax1.set_xlabel('Number of NaN Values')
    ax1.set_title('NaN Count by Column')
    ax1.grid(True, alpha=0.3)

# 2. NaN Percentage by Column
ax2 = plt.subplot(3, 2, 2)
if len(columns_with_nan_sorted) > 0:
    ax2.barh(range(len(columns_with_nan_sorted)), columns_with_nan_sorted['NaN_Percentage'], 
             color='orange', alpha=0.7)
    ax2.set_yticks(range(len(columns_with_nan_sorted)))
    ax2.set_yticklabels(columns_with_nan_sorted['Column'], fontsize=9)
    ax2.set_xlabel('Percentage of NaN Values (%)')
    ax2.set_title('NaN Percentage by Column')
    ax2.grid(True, alpha=0.3)

# 3. NaN Values Over Time (for columns with most NaN)
ax3 = plt.subplot(3, 2, 3)
if len(columns_with_nan) > 0:
    top_nan_cols = columns_with_nan.head(5)['Column'].tolist()
    for col in top_nan_cols:
        # Calculate rolling NaN rate
        nan_rate = games[col].isna().rolling(100).mean() * 100
        ax3.plot(range(len(games)), nan_rate, label=col, linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Game Index')
    ax3.set_ylabel('NaN Rate (%) - 100 game rolling')
    ax3.set_title('NaN Rate Over Time (Top 5 columns)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

# 4. NaN Heatmap by Season
ax4 = plt.subplot(3, 2, 4)
if len(columns_with_nan) > 0:
    # Create matrix: seasons × columns with NaN
    seasons = sorted(games['Season'].unique())
    nan_cols = columns_with_nan.head(15)['Column'].tolist()  # Top 15
    
    nan_matrix = []
    for season in seasons:
        season_games = games[games['Season'] == season]
        season_nan_pcts = []
        for col in nan_cols:
            pct = season_games[col].isna().sum() / len(season_games) * 100
            season_nan_pcts.append(pct)
        nan_matrix.append(season_nan_pcts)
    
    nan_matrix = np.array(nan_matrix).T
    
    im = ax4.imshow(nan_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=100)
    ax4.set_yticks(range(len(nan_cols)))
    ax4.set_yticklabels(nan_cols, fontsize=8)
    ax4.set_xticks(range(len(seasons)))
    ax4.set_xticklabels([int(s) for s in seasons], rotation=45, fontsize=7)
    ax4.set_xlabel('Season')
    ax4.set_title('NaN Percentage Heatmap by Season')
    plt.colorbar(im, ax=ax4, label='NaN %')

# 5. Data Completeness Score Over Time
ax5 = plt.subplot(3, 2, 5)
# Calculate completeness score (% of non-NaN values across all columns)
completeness = (1 - games.isna().mean(axis=1)) * 100
completeness_rolling = completeness.rolling(100).mean()
ax5.plot(range(len(games)), completeness_rolling, linewidth=2, color='green')
ax5.set_xlabel('Game Index')
ax5.set_ylabel('Data Completeness (%)')
ax5.set_title('Overall Data Completeness (100-game rolling average)')
ax5.axhline(100, color='gray', linestyle='--', linewidth=1, label='100% complete')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. NaN Count by Season
ax6 = plt.subplot(3, 2, 6)
nan_by_season = []
for season in sorted(games['Season'].unique()):
    season_games = games[games['Season'] == season]
    total_nan = season_games.isna().sum().sum()
    nan_by_season.append(total_nan)

ax6.bar(range(len(seasons)), nan_by_season, color='purple', alpha=0.7)
ax6.set_xticks(range(len(seasons)))
ax6.set_xticklabels([int(s) for s in seasons], rotation=45, fontsize=8)
ax6.set_xlabel('Season')
ax6.set_ylabel('Total NaN Count')
ax6.set_title('Total NaN Values by Season')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('games_nan_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: games_nan_analysis.png")

# Save detailed report to CSV
nan_summary.to_csv('games_nan_report.csv', index=False)
print(f"✓ Saved detailed report to: games_nan_report.csv")

# Identify patterns
print(f"\n🔍 PATTERN ANALYSIS:")

# Check if NaN values cluster in specific date ranges
if len(columns_with_nan) > 0:
    print(f"\n  Checking for temporal clustering...")
    for col in columns_with_nan.head(5)['Column'].tolist():
        nan_indices = games[games[col].isna()].index.tolist()
        if len(nan_indices) > 0:
            # Check if NaN values are consecutive
            consecutive_groups = []
            current_group = [nan_indices[0]]
            
            for i in range(1, len(nan_indices)):
                if nan_indices[i] - nan_indices[i-1] == 1:
                    current_group.append(nan_indices[i])
                else:
                    if len(current_group) > 10:  # Only report groups of 10+
                        consecutive_groups.append(current_group)
                    current_group = [nan_indices[i]]
            
            if len(current_group) > 10:
                consecutive_groups.append(current_group)
            
            if consecutive_groups:
                print(f"\n  {col}:")
                for group in consecutive_groups:
                    start_date = games.loc[group[0], 'Date'].date()
                    end_date = games.loc[group[-1], 'Date'].date()
                    print(f"    Consecutive NaN block: indices {group[0]}-{group[-1]} ({start_date} to {end_date})")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
