"""
Strategy Comparison Script

Compare your original betting strategy with Pepe's betting strategy
using the same ensemble predictions.
"""

import pandas as pd
import numpy as np
from model import Model as OriginalModel
from model_hybrid import Model as HybridModel
from environment import Environment


def compare_strategies(validation_days=365):
    """
    Compare original vs Pepe's betting strategy on validation data
    
    Args:
        validation_days: Number of days to use for validation
    """
    print("="*60)
    print("BETTING STRATEGY COMPARISON")
    print("="*60)
    print()
    
    # Load data
    print("Loading data...")
    games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])
    
    # Get validation split
    from datetime import timedelta
    all_dates = sorted(games['Date'].unique())
    val_end_date = all_dates[-1]
    train_end_date = val_end_date - timedelta(days=validation_days)
    
    print(f"Training period: up to {train_end_date.date()}")
    print(f"Validation period: {train_end_date.date()} to {val_end_date.date()}")
    print()
    
    # Test 1: Your original strategy
    print("-"*60)
    print("Testing YOUR ORIGINAL STRATEGY")
    print("-"*60)
    
    model_original = OriginalModel()
    env_original = Environment(
        games=games.copy(),
        model=model_original,
        start_date=None,
        end_date=val_end_date,
        init_bankroll=1000.0,
        min_bet=5,
        max_bet=100
    )
    
    evaluation_original = env_original.run()
    final_bankroll_original = env_original.bankroll
    profit_original = final_bankroll_original - 1000.0
    roi_original = (profit_original / 1000.0) * 100
    
    print(f"Final Bankroll: ${final_bankroll_original:.2f}")
    print(f"Profit: ${profit_original:.2f}")
    print(f"ROI: {roi_original:.2f}%")
    print()
    
    # Test 2: Pepe's strategy with your predictions
    print("-"*60)
    print("Testing PEPE'S STRATEGY (with your predictions)")
    print("-"*60)
    
    model_hybrid = HybridModel()
    env_hybrid = Environment(
        games=games.copy(),
        model=model_hybrid,
        start_date=None,
        end_date=val_end_date,
        init_bankroll=1000.0,
        min_bet=5,
        max_bet=100
    )
    
    evaluation_hybrid = env_hybrid.run()
    final_bankroll_hybrid = env_hybrid.bankroll
    profit_hybrid = final_bankroll_hybrid - 1000.0
    roi_hybrid = (profit_hybrid / 1000.0) * 100
    
    print(f"Final Bankroll: ${final_bankroll_hybrid:.2f}")
    print(f"Profit: ${profit_hybrid:.2f}")
    print(f"ROI: {roi_hybrid:.2f}%")
    print()
    
    # Comparison
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print()
    
    print(f"{'Strategy':<30} {'Bankroll':<15} {'Profit':<15} {'ROI':<10}")
    print("-"*60)
    print(f"{'Your Original':<30} ${final_bankroll_original:<14.2f} ${profit_original:<14.2f} {roi_original:<9.2f}%")
    print(f"{'Pepe (Hybrid)':<30} ${final_bankroll_hybrid:<14.2f} ${profit_hybrid:<14.2f} {roi_hybrid:<9.2f}%")
    print()
    
    # Winner
    if profit_hybrid > profit_original:
        improvement = ((profit_hybrid - profit_original) / abs(profit_original)) * 100 if profit_original != 0 else float('inf')
        print(f"🏆 WINNER: Pepe's Strategy")
        print(f"   Improvement: ${profit_hybrid - profit_original:.2f} ({improvement:+.2f}%)")
    elif profit_original > profit_hybrid:
        improvement = ((profit_original - profit_hybrid) / abs(profit_hybrid)) * 100 if profit_hybrid != 0 else float('inf')
        print(f"🏆 WINNER: Your Original Strategy")
        print(f"   Improvement: ${profit_original - profit_hybrid:.2f} ({improvement:+.2f}%)")
    else:
        print("🤝 TIE: Both strategies performed equally")
    
    print()
    print("="*60)
    
    return {
        'original': {
            'bankroll': final_bankroll_original,
            'profit': profit_original,
            'roi': roi_original
        },
        'hybrid': {
            'bankroll': final_bankroll_hybrid,
            'profit': profit_hybrid,
            'roi': roi_hybrid
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare betting strategies')
    parser.add_argument('--validation-days', type=int, default=365,
                       help='Number of days for validation (default: 365)')
    
    args = parser.parse_args()
    
    results = compare_strategies(validation_days=args.validation_days)
