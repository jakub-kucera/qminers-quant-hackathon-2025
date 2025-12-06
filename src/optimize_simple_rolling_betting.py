"""
Optimize Betting Strategy for Simple Rolling Window Model

This script uses Optuna to find optimal betting parameters while keeping
the prediction model fixed.

Optimized parameters:
    - min_edge: Minimum edge required to place bet
    - max_kelly_fraction: Maximum Kelly fraction (safety cap)
    - kelly_multiplier: Multiplier on Kelly bet size
    - normalize_odds: Whether to normalize bookmaker odds to remove margin

Usage:
    python optimize_simple_rolling_betting.py --n-trials 100 --n-jobs 4
"""

import argparse
import pandas as pd
import optuna
from datetime import timedelta

from model_simple_rolling import Model
from environment import Environment


def objective(trial):
    """Optuna objective function"""
    
    # Load data
    games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])
    
    # Note: Wider ranges to accommodate both simple and normalized odds approaches
    min_edge = trial.suggest_float('min_edge', 0.0, 0.20)  # 0-12% (normalized needs higher)
    max_kelly_fraction = trial.suggest_float('max_kelly_fraction', 0.05, 0.50)  # 5-30% max Kelly
    kelly_multiplier = trial.suggest_float('kelly_multiplier', 0.2, 2)  # 0.2x-1.5x Kelly
    normalize_odds = trial.suggest_categorical('normalize_odds', [True, False])  # Test both
    min_probability = trial.suggest_float('min_probability', 0.45, 0.95)  # 50-95% min probability

    
    # Create model with suggested parameters
    model = Model(
        # Fixed prediction parameters
        rolling_window_short=5,
        rolling_window_med=10,
        rolling_window_long=20,
        required_games=100,
        retrain_freq=50,
        # Optimized betting parameters
        min_edge=min_edge,
        max_kelly_fraction=max_kelly_fraction,
        kelly_multiplier=kelly_multiplier,
        normalize_odds=normalize_odds,
        min_probability=min_probability,
    )
    
    # Run evaluation
    env = Environment(
        games=games.copy(),
        model=model,
        init_bankroll=1000.0,
        min_bet=5,
        max_bet=100
    )
    
    try:
        env.run()
        profit = env.bankroll - 1000.0
        return profit
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description='Optimize betting strategy parameters')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--study-name', type=str, default='simple_rolling_betting_opt',
                       help='Name of the optimization study')
    
    args = parser.parse_args()
    
    # Setup storage for parallel execution
    storage = None
    if args.n_jobs != 1:
        storage = f'sqlite:///{args.study_name}.db'
        print(f"Using SQLite storage: {storage}")
        
        # Optimize SQLite for parallel writes
        import sqlite3
        db_path = storage.replace('sqlite:///', '')
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.close()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize profit
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization
    print(f"\nStarting optimization with {args.n_trials} trials...")
    print(f"Parallel workers: {args.n_jobs}")
    print(f"Objective: Maximize profit\n")
    
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nBest trial:")
    print(f"  Profit: ${study.best_value:.2f}")
    print(f"  ROI: {(study.best_value / 1000 * 100):.2f}%")
    
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print(f"\nTo use these parameters:")
    print(f"""
model = Model(
    min_edge={study.best_params['min_edge']:.4f},
    max_kelly_fraction={study.best_params['max_kelly_fraction']:.4f},
    kelly_multiplier={study.best_params['kelly_multiplier']:.4f},
    normalize_odds={study.best_params['normalize_odds']}
    min_probability={study.best_params['min_probability']:.4f}
)
""")


if __name__ == '__main__':
    main()
