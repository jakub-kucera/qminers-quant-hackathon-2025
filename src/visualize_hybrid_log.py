"""
Visualize Hybrid Model Run

This script analyzes 'hybrid_model_log.csv' and creates visualizations:
- Bankroll over time
- Daily Betting Activity (Number of bets & Exposure)
- Ensemble Coefficient Stability
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Load the log file
log_file = "hybrid_model_log.csv"
try:
    df = pd.read_csv(log_file, parse_dates=['date'])
except FileNotFoundError:
    print(f"Error: {log_file} not found. Run example_run_hybrid_with_logging.py first.")
    exit(1)

print(f"Loaded {len(df)} log entries")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Bankroll range: ${df['bankroll'].min():.2f} to ${df['bankroll'].max():.2f}")

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Bankroll Over Time with Moving Average
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df['date'], df['bankroll'], 'b-', linewidth=1.5, label='Bankroll', alpha=0.8)

# Add moving average trend
ma_window = 30 # 30 days
df['bankroll_ma'] = df['bankroll'].rolling(window=ma_window).mean()
ax1.plot(df['date'], df['bankroll_ma'], 'r--', linewidth=2, label=f'{ma_window}-Day MA')

ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_ylabel('Bankroll ($)')
ax1.set_title('Bankroll Growth Over Time')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Filter for days where betting actually happened (active seasons)
betting_days = df[df['n_bets_today'] > 0].copy()

# 2. Daily Betting Activity (Bar Chart)
ax2 = plt.subplot(3, 1, 2)
if not betting_days.empty:
    ax2.bar(betting_days['date'], betting_days['n_bets_today'], color='teal', alpha=0.6, label='Number of Bets', width=2.0)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(betting_days['date'], betting_days['total_exposure_today'], 'g-', linewidth=0.8, alpha=0.4, label='Total Daily Exposure')
    
    ax2.set_ylabel('Number of Bets (Bars)')
    ax2_twin.set_ylabel('Total Exposure $ (Line)', color='green')
    ax2.set_title('Daily Betting Activity')
    
    # Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax2.text(0.5, 0.5, 'No Bets Placed Yet', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

# 3. Ensemble Coefficients Evolution
ax3 = plt.subplot(3, 1, 3)

# Identify coefficient columns
coef_cols = [c for c in df.columns if c.startswith('ensemble_coef_')]

if coef_cols and not betting_days.empty:
    # Only plot if we have data
    valid_coefs = df[coef_cols].replace(0, np.nan).dropna(how='all')
    
    # Plot top 5 coefficients by max magnitude
    if not valid_coefs.empty:
        max_vals = valid_coefs.abs().max()
        top_cols = max_vals.nlargest(5).index
        
        for col in top_cols:
            # Smooth lines for readability
            smoothed = df[col].rolling(window=10).mean()
            ax3.plot(df['date'], smoothed, linewidth=2, label=col)
            
        ax3.set_ylabel('Coefficient Value (Smoothed)')
        ax3.set_title('Ensemble Model Weights Evolution (Top 5)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Coefficients are all zero (Model not trained yet)', horizontalalignment='center', transform=ax3.transAxes)

plt.tight_layout()
output_file = 'hybrid_model_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_file}")

# Print summary
final_bankroll = df['bankroll'].iloc[-1]
profit = final_bankroll - 1000
roi = (profit / 1000) * 100
total_bets = df['n_bets_today'].sum()
avg_bet_size = df[df['total_exposure_today'] > 0]['total_exposure_today'].sum() / total_bets if total_bets > 0 else 0

print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Final Bankroll:    ${final_bankroll:,.2f}")
print(f"Total Profit:      ${profit:,.2f}")
print(f"ROI:               {roi:.2f}%")
print(f"Total Bets Placed: {total_bets:.0f}")
print(f"Avg Bet Size:      ${avg_bet_size:.2f}")
if not betting_days.empty:
    print(f"Active Betting Days: {len(betting_days)}")
print("="*60)
