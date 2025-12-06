"""
Visualize Ensemble Weights Over Time

This script reads the ensemble weights log and creates visualizations showing:
1. How each sub-model's importance changes over time
2. Bankroll progression
3. Feature weights for each outcome class (Away/Draw/Home)

Usage:
    python visualize_ensemble_weights.py [--log-file ensemble_weights_log.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_weights_log(log_file):
    """Load the ensemble weights log CSV"""
    df = pd.read_csv(log_file)
    return df


def aggregate_weights_by_model(df):
    """
    Aggregate feature weights by sub-model for easier visualization
    
    Returns dict with model names as keys and DataFrames of weights over time
    """
    # Define which columns belong to which sub-model
    model_features = {
        'Pythagorean': ['pyth_home_win_pct', 'pyth_away_win_pct'],
        'FourFactor': ['four_factor_diff'],
        'GradientDescent': ['gradient_descent_diff'],
        'Exhaustion': ['exhaustion_factor'],
        'ELO': ['elo_diff'],
        'Poisson': ['poisson_goal_diff'],
        'RollingShort': [
            'roll_short_goal_diff', 'roll_short_ga_diff', 'roll_short_sog_diff',
            'roll_short_pp_diff', 'roll_short_fo_diff', 'roll_short_win_diff', 'roll_short_sh_diff'
        ],
        'RollingMed': [
            'roll_med_goal_diff', 'roll_med_ga_diff', 'roll_med_sog_diff',
            'roll_med_pp_diff', 'roll_med_fo_diff', 'roll_med_win_diff', 'roll_med_sh_diff'
        ],
        'RollingLong': [
            'roll_long_goal_diff', 'roll_long_ga_diff', 'roll_long_sog_diff',
            'roll_long_pp_diff', 'roll_long_fo_diff', 'roll_long_win_diff', 'roll_long_sh_diff'
        ],
    }
    
    aggregated = {}
    
    for model_name, features in model_features.items():
        # For each class (away, draw, home), sum absolute weights of all features in this model
        model_data = {
            'game_num': df['game_num'],
            'date': pd.to_datetime(df['date']),
            'bankroll': df['bankroll']
        }
        
        for outcome in ['away', 'draw', 'home']:
            # Get columns for this outcome and these features
            cols = [f'{outcome}_{feat}' for feat in features]
            # Sum absolute values to get total importance
            model_data[f'{outcome}_importance'] = df[cols].abs().sum(axis=1)
        
        aggregated[model_name] = pd.DataFrame(model_data)
    
    return aggregated


def plot_bankroll(df, output_dir):
    """Plot bankroll over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['game_num'], df['bankroll'], linewidth=2, color='green')
    ax.axhline(y=1000, color='red', linestyle='--', label='Initial Bankroll', alpha=0.5)
    ax.axhline(y=1200, color='orange', linestyle='--', label='Stop Threshold', alpha=0.5)
    
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Bankroll ($)', fontsize=12)
    ax.set_title('Bankroll Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bankroll_over_time.png', dpi=150)
    print(f"Saved: {output_dir / 'bankroll_over_time.png'}")
    plt.close()


def plot_model_importance_by_outcome(aggregated, output_dir):
    """Plot each sub-model's importance for each outcome over time"""
    outcomes = ['away', 'draw', 'home']
    colors = {'away': 'blue', 'draw': 'gray', 'home': 'red'}
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Sub-Model Importance by Outcome Over Time', fontsize=16, fontweight='bold')
    
    for idx, (model_name, data) in enumerate(aggregated.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for outcome in outcomes:
            ax.plot(data['game_num'], data[f'{outcome}_importance'],
                   label=outcome.capitalize(), color=colors[outcome], alpha=0.7)
        
        ax.set_title(model_name, fontweight='bold')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Total Absolute Weight')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_importance_by_outcome.png', dpi=150)
    print(f"Saved: {output_dir / 'model_importance_by_outcome.png'}")
    plt.close()


def plot_total_model_importance(aggregated, output_dir):
    """Plot total importance of each sub-model (summed across all outcomes)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for model_name, data in aggregated.items():
        # Sum importance across all outcomes
        total_importance = (data['away_importance'] + 
                          data['draw_importance'] + 
                          data['home_importance'])
        ax.plot(data['game_num'], total_importance, label=model_name, linewidth=2)
    
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Total Importance (Sum of Absolute Weights)', fontsize=12)
    ax.set_title('Sub-Model Total Importance Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'total_model_importance.png', dpi=150)
    print(f"Saved: {output_dir / 'total_model_importance.png'}")
    plt.close()


def plot_correlation_with_bankroll(aggregated, output_dir):
    """Plot correlation between model importance and bankroll changes"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Model Importance vs Bankroll', fontsize=16, fontweight='bold')
    
    for idx, (model_name, data) in enumerate(aggregated.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Calculate total importance
        total_importance = (data['away_importance'] + 
                          data['draw_importance'] + 
                          data['home_importance'])
        
        # Plot on dual axis
        ax2 = ax.twinx()
        
        ax.plot(data['game_num'], total_importance, 
               color='blue', label='Importance', linewidth=2)
        ax2.plot(data['game_num'], data['bankroll'], 
                color='green', label='Bankroll', linewidth=2, alpha=0.7)
        
        ax.set_title(model_name, fontweight='bold')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Importance', color='blue')
        ax2.set_ylabel('Bankroll ($)', color='green')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'importance_vs_bankroll.png', dpi=150)
    print(f"Saved: {output_dir / 'importance_vs_bankroll.png'}")
    plt.close()


def plot_feature_weights_heatmap(df, output_dir, outcome='home', top_n=15):
    """Plot heatmap of top N feature weights over time for a specific outcome"""
    # Get all feature columns for this outcome
    feature_cols = [col for col in df.columns if col.startswith(f'{outcome}_') and col != f'{outcome}_importance']
    
    # Calculate mean absolute weight for each feature
    mean_weights = df[feature_cols].abs().mean()
    top_features = mean_weights.nlargest(top_n).index.tolist()
    
    # Create heatmap data
    heatmap_data = df[top_features].T
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                   vmin=-heatmap_data.abs().max().max(), 
                   vmax=heatmap_data.abs().max().max())
    
    # Set ticks
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace(f'{outcome}_', '') for f in top_features])
    
    # Show every 50th game number
    x_ticks = range(0, len(df), max(1, len(df) // 20))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([df.iloc[i]['game_num'] for i in x_ticks], rotation=45)
    
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Weights for {outcome.capitalize()} Outcome Over Time', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Weight Value')
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_weights_heatmap_{outcome}.png', dpi=150)
    print(f"Saved: {output_dir / f'feature_weights_heatmap_{outcome}.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ensemble weights over time')
    parser.add_argument('--log-file', type=str, default='ensemble_weights_log.csv',
                       help='Path to ensemble weights log CSV')
    parser.add_argument('--output-dir', type=str, default='ensemble_analysis',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading weights from: {args.log_file}")
    df = load_weights_log(args.log_file)
    
    print(f"Loaded {len(df)} game records")
    print(f"Bankroll range: ${df['bankroll'].min():.2f} - ${df['bankroll'].max():.2f}")
    
    # Aggregate weights by model
    print("\nAggregating weights by sub-model...")
    aggregated = aggregate_weights_by_model(df)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_bankroll(df, output_dir)
    plot_model_importance_by_outcome(aggregated, output_dir)
    plot_total_model_importance(aggregated, output_dir)
    plot_correlation_with_bankroll(aggregated, output_dir)
    
    # Heatmaps for each outcome
    for outcome in ['away', 'draw', 'home']:
        plot_feature_weights_heatmap(df, output_dir, outcome=outcome, top_n=15)
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
