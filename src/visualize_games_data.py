"""
NHL Games Data Visualization

This script creates comprehensive visualizations of the games.csv dataset,
showing trends, distributions, and patterns in the NHL data over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 15)

# Load data
print("Loading games data...")
games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])

print(f"Loaded {len(games)} games")
print(f"Date range: {games['Date'].min()} to {games['Date'].max()}")
print(f"Seasons: {games['Season'].min()} to {games['Season'].max()}")

# Create figure with subplots
fig = plt.figure(figsize=(24, 18))

# 1. Games per Season
ax1 = plt.subplot(4, 3, 1)
games_per_season = games.groupby('Season').size()
ax1.bar(games_per_season.index, games_per_season.values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Games')
ax1.set_title('Games per Season')
ax1.grid(True, alpha=0.3)

# 2. Home vs Away Wins
ax2 = plt.subplot(4, 3, 2)
home_wins = games['H'].sum()
away_wins = games['A'].sum()
draws = games['D'].sum()
ax2.pie([home_wins, away_wins, draws], 
        labels=['Home Wins', 'Away Wins', 'Draws'],
        autopct='%1.1f%%',
        colors=['green', 'red', 'gray'],
        startangle=90)
ax2.set_title(f'Game Outcomes (Total: {len(games)})')

# 3. Goals Distribution
ax3 = plt.subplot(4, 3, 3)
total_goals = games['HS'] + games['AS']
ax3.hist(total_goals, bins=30, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(total_goals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {total_goals.mean():.2f}')
ax3.set_xlabel('Total Goals per Game')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Total Goals per Game')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Average Goals Over Time
ax4 = plt.subplot(4, 3, 4)
games['Total_Goals'] = games['HS'] + games['AS']
goals_by_season = games.groupby('Season')['Total_Goals'].mean()
ax4.plot(goals_by_season.index, goals_by_season.values, marker='o', linewidth=2, markersize=6, color='darkblue')
ax4.set_xlabel('Season')
ax4.set_ylabel('Average Goals per Game')
ax4.set_title('Average Goals per Game by Season')
ax4.grid(True, alpha=0.3)

# 5. Shots on Goal Distribution
ax5 = plt.subplot(4, 3, 5)
ax5.scatter(games['H_SOG'], games['A_SOG'], alpha=0.3, s=10, c='purple')
ax5.plot([0, 60], [0, 60], 'r--', linewidth=2, label='Equal shots')
ax5.set_xlabel('Home Shots on Goal')
ax5.set_ylabel('Away Shots on Goal')
ax5.set_title('Shots on Goal: Home vs Away')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Power Play Goals
ax6 = plt.subplot(4, 3, 6)
ppg_data = pd.DataFrame({
    'Home PPG': games['H_PPG'],
    'Away PPG': games['A_PPG']
})
ppg_data.boxplot(ax=ax6)
ax6.set_ylabel('Power Play Goals')
ax6.set_title('Power Play Goals Distribution')
ax6.grid(True, alpha=0.3)

# 7. Overtime/Shootout Games
ax7 = plt.subplot(4, 3, 7)
special_games = games['Special'].value_counts()
special_games = special_games[special_games.index != '']  # Remove empty
if len(special_games) > 0:
    ax7.bar(special_games.index, special_games.values, color=['gold', 'silver'], alpha=0.7)
    ax7.set_xlabel('Game Type')
    ax7.set_ylabel('Count')
    ax7.set_title('Overtime and Shootout Games')
    ax7.grid(True, alpha=0.3)

# 8. Penalty Minutes Distribution
ax8 = plt.subplot(4, 3, 8)
total_pim = games['H_PIM'] + games['A_PIM']
ax8.hist(total_pim, bins=40, color='red', alpha=0.6, edgecolor='black')
ax8.axvline(total_pim.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {total_pim.mean():.1f}')
ax8.set_xlabel('Total Penalty Minutes')
ax8.set_ylabel('Frequency')
ax8.set_title('Distribution of Penalty Minutes per Game')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Faceoff Win Percentage
ax9 = plt.subplot(4, 3, 9)
games['H_FO_pct'] = games['H_FO'] / (games['H_FO'] + games['A_FO'])
ax9.hist(games['H_FO_pct'].dropna(), bins=30, color='teal', alpha=0.7, edgecolor='black')
ax9.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50%')
ax9.set_xlabel('Home Faceoff Win %')
ax9.set_ylabel('Frequency')
ax9.set_title('Home Team Faceoff Win Percentage Distribution')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Home Advantage Over Time
ax10 = plt.subplot(4, 3, 10)
home_win_pct = games.groupby('Season')['H'].mean()
ax10.plot(home_win_pct.index, home_win_pct.values * 100, marker='o', linewidth=2, markersize=6, color='green')
ax10.axhline(50, color='red', linestyle='--', linewidth=2, label='50% (no advantage)')
ax10.set_xlabel('Season')
ax10.set_ylabel('Home Win Percentage')
ax10.set_title('Home Ice Advantage by Season')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Odds Distribution
ax11 = plt.subplot(4, 3, 11)
ax11.hist(games['OddsH'], bins=30, alpha=0.5, label='Home Odds', color='green', edgecolor='black')
ax11.hist(games['OddsA'], bins=30, alpha=0.5, label='Away Odds', color='red', edgecolor='black')
ax11.set_xlabel('Odds')
ax11.set_ylabel('Frequency')
ax11.set_title('Distribution of Betting Odds')
ax11.legend()
ax11.grid(True, alpha=0.3)

# 12. Scoring by Period
ax12 = plt.subplot(4, 3, 12)
period_goals = pd.DataFrame({
    'Period 1': games['H_P1'] + games['A_P1'],
    'Period 2': games['H_P2'] + games['A_P2'],
    'Period 3': games['H_P3'] + games['A_P3']
})
period_goals.mean().plot(kind='bar', ax=ax12, color=['lightblue', 'skyblue', 'steelblue'], alpha=0.7)
ax12.set_xlabel('Period')
ax12.set_ylabel('Average Goals')
ax12.set_title('Average Goals by Period')
ax12.set_xticklabels(ax12.get_xticklabels(), rotation=0)
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('games_data_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to: games_data_visualization.png")

# Print summary statistics
print("\n" + "="*60)
print("DATASET SUMMARY STATISTICS")
print("="*60)

print(f"\n📊 General Statistics:")
print(f"  Total games: {len(games):,}")
print(f"  Seasons covered: {games['Season'].nunique()}")
print(f"  Unique teams: {pd.concat([games['HID'], games['AID']]).nunique()}")
print(f"  Date range: {games['Date'].min().date()} to {games['Date'].max().date()}")

print(f"\n🏒 Game Outcomes:")
print(f"  Home wins: {home_wins:,} ({home_wins/len(games)*100:.1f}%)")
print(f"  Away wins: {away_wins:,} ({away_wins/len(games)*100:.1f}%)")
print(f"  Draws: {draws:,} ({draws/len(games)*100:.1f}%)")

print(f"\n⚽ Scoring Statistics:")
print(f"  Average goals per game: {total_goals.mean():.2f}")
print(f"  Average home goals: {games['HS'].mean():.2f}")
print(f"  Average away goals: {games['AS'].mean():.2f}")
print(f"  Highest scoring game: {total_goals.max()} goals")
print(f"  Lowest scoring game: {total_goals.min()} goals")

print(f"\n🎯 Shooting Statistics:")
print(f"  Average home SOG: {games['H_SOG'].mean():.1f}")
print(f"  Average away SOG: {games['A_SOG'].mean():.1f}")
print(f"  Average shooting % (home): {(games['HS'] / games['H_SOG'] * 100).mean():.1f}%")
print(f"  Average shooting % (away): {(games['AS'] / games['A_SOG'] * 100).mean():.1f}%")

print(f"\n⚡ Power Play Statistics:")
print(f"  Average home PPG: {games['H_PPG'].mean():.2f}")
print(f"  Average away PPG: {games['A_PPG'].mean():.2f}")
print(f"  Games with PP goals: {((games['H_PPG'] > 0) | (games['A_PPG'] > 0)).sum():,} ({((games['H_PPG'] > 0) | (games['A_PPG'] > 0)).sum()/len(games)*100:.1f}%)")

print(f"\n🚨 Penalty Statistics:")
print(f"  Average penalty minutes: {total_pim.mean():.1f}")
print(f"  Average home PIM: {games['H_PIM'].mean():.1f}")
print(f"  Average away PIM: {games['A_PIM'].mean():.1f}")

print(f"\n🎲 Betting Odds:")
print(f"  Average home odds: {games['OddsH'].mean():.2f}")
print(f"  Average away odds: {games['OddsA'].mean():.2f}")
print(f"  Median home odds: {games['OddsH'].median():.2f}")
print(f"  Median away odds: {games['OddsA'].median():.2f}")

# Special games
ot_games = (games['Special'] == 'OT').sum()
so_games = (games['Special'] == 'SO').sum()
print(f"\n⏱️  Special Games:")
print(f"  Overtime games: {ot_games:,} ({ot_games/len(games)*100:.1f}%)")
print(f"  Shootout games: {so_games:,} ({so_games/len(games)*100:.1f}%)")

print("\n" + "="*60)
print("Visualization complete! Check games_data_visualization.png")
print("="*60)
