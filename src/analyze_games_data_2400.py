"""
Analyze Raw Game Data Changes Around Game 2400

This script analyzes the actual game data (not model performance) around game 2400
to identify what changed in the underlying NHL data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (24, 16)

# Configuration
CENTER_GAME = 2400
WINDOW = 200
START_IDX = CENTER_GAME - WINDOW
END_IDX = CENTER_GAME + WINDOW

print(f"Analyzing games data from index {START_IDX} to {END_IDX}")

# Load games data
print("\nLoading games.csv...")
games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

# Filter to window
games_window = games.iloc[START_IDX:END_IDX+1].copy()

print(f"Loaded {len(games_window)} games")
print(f"Date range: {games_window['Date'].min().date()} to {games_window['Date'].max().date()}")
print(f"Seasons: {sorted(games_window['Season'].unique())}")

# Add derived columns
games_window['Total_Goals'] = games_window['HS'] + games_window['AS']
games_window['Goal_Diff'] = abs(games_window['HS'] - games_window['AS'])
games_window['Total_SOG'] = games_window['H_SOG'] + games_window['A_SOG']
games_window['Total_PIM'] = games_window['H_PIM'] + games_window['A_PIM']
games_window['Total_PPG'] = games_window['H_PPG'] + games_window['A_PPG']

# Create visualizations
fig = plt.figure(figsize=(24, 18))

# 1. Total Goals Over Time
ax1 = plt.subplot(4, 3, 1)
ax1.plot(range(len(games_window)), games_window['Total_Goals'], alpha=0.5, linewidth=1)
ax1.plot(range(len(games_window)), games_window['Total_Goals'].rolling(20).mean(), 
         linewidth=2, color='red', label='20-game average')
ax1.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax1.set_xlabel('Game Index (relative)')
ax1.set_ylabel('Total Goals')
ax1.set_title(f'Total Goals per Game (Games {START_IDX}-{END_IDX})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Home Win Rate
ax2 = plt.subplot(4, 3, 2)
home_win_rate = games_window['H'].rolling(50).mean() * 100
ax2.plot(range(len(games_window)), home_win_rate, linewidth=2, color='green')
ax2.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% (no advantage)')
ax2.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax2.set_xlabel('Game Index (relative)')
ax2.set_ylabel('Home Win Rate (%)')
ax2.set_title('Home Win Rate (50-game rolling average)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Shots on Goal
ax3 = plt.subplot(4, 3, 3)
ax3.plot(range(len(games_window)), games_window['Total_SOG'].rolling(20).mean(), 
         linewidth=2, color='blue')
ax3.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax3.set_xlabel('Game Index (relative)')
ax3.set_ylabel('Total Shots on Goal')
ax3.set_title('Average Shots on Goal (20-game rolling)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Penalty Minutes
ax4 = plt.subplot(4, 3, 4)
ax4.plot(range(len(games_window)), games_window['Total_PIM'].rolling(20).mean(), 
         linewidth=2, color='red')
ax4.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax4.set_xlabel('Game Index (relative)')
ax4.set_ylabel('Total Penalty Minutes')
ax4.set_title('Average Penalty Minutes (20-game rolling)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Power Play Goals
ax5 = plt.subplot(4, 3, 5)
ax5.plot(range(len(games_window)), games_window['Total_PPG'].rolling(20).mean(), 
         linewidth=2, color='orange')
ax5.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax5.set_xlabel('Game Index (relative)')
ax5.set_ylabel('Total PP Goals')
ax5.set_title('Average Power Play Goals (20-game rolling)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Overtime/Shootout Games
ax6 = plt.subplot(4, 3, 6)
games_window['Is_OT_SO'] = games_window['Special'].isin(['OT', 'SO']).astype(int)
ot_so_rate = games_window['Is_OT_SO'].rolling(50).mean() * 100
ax6.plot(range(len(games_window)), ot_so_rate, linewidth=2, color='purple')
ax6.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax6.set_xlabel('Game Index (relative)')
ax6.set_ylabel('OT/SO Rate (%)')
ax6.set_title('Overtime/Shootout Game Rate (50-game rolling)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Betting Odds - Home
ax7 = plt.subplot(4, 3, 7)
ax7.plot(range(len(games_window)), games_window['OddsH'].rolling(20).mean(), 
         linewidth=2, color='green', label='Home Odds')
ax7.plot(range(len(games_window)), games_window['OddsA'].rolling(20).mean(), 
         linewidth=2, color='red', label='Away Odds')
ax7.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax7.set_xlabel('Game Index (relative)')
ax7.set_ylabel('Average Odds')
ax7.set_title('Betting Odds (20-game rolling average)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Goal Differential Distribution
ax8 = plt.subplot(4, 3, 8)
before_center = games_window.iloc[:WINDOW]
after_center = games_window.iloc[WINDOW:]
ax8.hist(before_center['Goal_Diff'], bins=15, alpha=0.5, label=f'Before {CENTER_GAME}', color='blue')
ax8.hist(after_center['Goal_Diff'], bins=15, alpha=0.5, label=f'After {CENTER_GAME}', color='red')
ax8.set_xlabel('Goal Differential')
ax8.set_ylabel('Frequency')
ax8.set_title('Goal Differential Distribution: Before vs After')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Shooting Percentage
ax9 = plt.subplot(4, 3, 9)
games_window['Shooting_Pct'] = (games_window['Total_Goals'] / games_window['Total_SOG'] * 100)
ax9.plot(range(len(games_window)), games_window['Shooting_Pct'].rolling(20).mean(), 
         linewidth=2, color='teal')
ax9.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax9.set_xlabel('Game Index (relative)')
ax9.set_ylabel('Shooting % ')
ax9.set_title('Average Shooting Percentage (20-game rolling)')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Season Timeline
ax10 = plt.subplot(4, 3, 10)
season_changes = games_window['Season'].diff() != 0
season_change_indices = games_window.index[season_changes].tolist()

for season in games_window['Season'].unique():
    season_mask = games_window['Season'] == season
    season_indices = np.where(season_mask)[0]
    if len(season_indices) > 0:
        ax10.axvspan(season_indices[0], season_indices[-1], alpha=0.3, 
                     label=f'Season {int(season)}')

ax10.axvline(WINDOW, color='darkred', linestyle='--', linewidth=3, label=f'Game {CENTER_GAME}')
ax10.set_xlabel('Game Index (relative)')
ax10.set_ylabel('Season')
ax10.set_title('Season Timeline')
ax10.legend(loc='best', fontsize=8)
ax10.grid(True, alpha=0.3)

# 11. Data Completeness
ax11 = plt.subplot(4, 3, 11)
# Check for missing data
missing_cols = ['H_SOG', 'A_SOG', 'H_PPG', 'A_PPG', 'H_PIM', 'A_PIM']
missing_pct = []
for col in missing_cols:
    missing_pct.append(games_window[col].isna().sum() / len(games_window) * 100)

ax11.bar(range(len(missing_cols)), missing_pct, color='orange', alpha=0.7)
ax11.set_xticks(range(len(missing_cols)))
ax11.set_xticklabels(missing_cols, rotation=45, ha='right')
ax11.set_ylabel('Missing Data (%)')
ax11.set_title('Data Completeness Check')
ax11.grid(True, alpha=0.3)

# 12. Faceoff Win Percentage
ax12 = plt.subplot(4, 3, 12)
games_window['FO_Total'] = games_window['H_FO'] + games_window['A_FO']
ax12.plot(range(len(games_window)), games_window['FO_Total'].rolling(20).mean(), 
          linewidth=2, color='brown')
ax12.axvline(WINDOW, color='darkred', linestyle='--', linewidth=2, label=f'Game {CENTER_GAME}')
ax12.set_xlabel('Game Index (relative)')
ax12.set_ylabel('Total Faceoffs')
ax12.set_title('Average Total Faceoffs (20-game rolling)')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'games_data_analysis_{CENTER_GAME}.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: games_data_analysis_{CENTER_GAME}.png")

# Print detailed statistics
print("\n" + "="*80)
print(f"RAW GAME DATA ANALYSIS AROUND GAME {CENTER_GAME}")
print("="*80)

before = games_window.iloc[:WINDOW]
after = games_window.iloc[WINDOW:]

print(f"\n📊 BEFORE Game {CENTER_GAME} (games {START_IDX}-{CENTER_GAME-1}):")
print(f"  Date range: {before['Date'].min().date()} to {before['Date'].max().date()}")
print(f"  Seasons: {sorted(before['Season'].unique())}")
print(f"  Average goals: {before['Total_Goals'].mean():.2f}")
print(f"  Average SOG: {before['Total_SOG'].mean():.1f}")
print(f"  Average PIM: {before['Total_PIM'].mean():.1f}")
print(f"  Home win rate: {before['H'].mean()*100:.1f}%")
print(f"  OT/SO rate: {before['Is_OT_SO'].mean()*100:.1f}%")

print(f"\n📊 AFTER Game {CENTER_GAME} (games {CENTER_GAME}-{END_IDX}):")
print(f"  Date range: {after['Date'].min().date()} to {after['Date'].max().date()}")
print(f"  Seasons: {sorted(after['Season'].unique())}")
print(f"  Average goals: {after['Total_Goals'].mean():.2f}")
print(f"  Average SOG: {after['Total_SOG'].mean():.1f}")
print(f"  Average PIM: {after['Total_PIM'].mean():.1f}")
print(f"  Home win rate: {after['H'].mean()*100:.1f}%")
print(f"  OT/SO rate: {after['Is_OT_SO'].mean()*100:.1f}%")

print(f"\n📈 CHANGES:")
print(f"  Goals: {after['Total_Goals'].mean() - before['Total_Goals'].mean():+.2f}")
print(f"  SOG: {after['Total_SOG'].mean() - before['Total_SOG'].mean():+.1f}")
print(f"  PIM: {after['Total_PIM'].mean() - before['Total_PIM'].mean():+.1f}")
print(f"  Home win rate: {(after['H'].mean() - before['H'].mean())*100:+.1f}%")
print(f"  OT/SO rate: {(after['Is_OT_SO'].mean() - before['Is_OT_SO'].mean())*100:+.1f}%")

# Check for season boundary
if before['Season'].iloc[-1] != after['Season'].iloc[0]:
    print(f"\n⚠️  SEASON BOUNDARY DETECTED!")
    print(f"  Last game before: Season {int(before['Season'].iloc[-1])}")
    print(f"  First game after: Season {int(after['Season'].iloc[0])}")
    print(f"  This is likely the cause of data changes!")

# Check for missing data spikes
print(f"\n🔍 DATA QUALITY:")
for col in missing_cols:
    before_missing = before[col].isna().sum()
    after_missing = after[col].isna().sum()
    if before_missing > 0 or after_missing > 0:
        print(f"  {col}: {before_missing} missing before, {after_missing} missing after")

print("\n" + "="*80)
print(f"Analysis complete! Check games_data_analysis_{CENTER_GAME}.png")
print("="*80)
