"""
Betting Strategy Hyperparameter Optimization

This script tunes betting strategy parameters (bet_threshold, bet_size_base, bet_kelly_fraction)
to maximize final bankroll while keeping prediction model hyperparameters fixed.

Usage:
    python3 src/optimize_betting.py --n-trials 100 --n-jobs 4
    python3 src/optimize_betting.py --n-trials 50  # Sequential execution
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import optuna

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import Model
from src.environment import Environment


class BettingOptimizer:
    """Optimize betting strategy parameters to maximize bankroll"""
    
    def __init__(self, validation_period_days=365):
        """
        Args:
            validation_period_days: Number of days to use for validation
        """
        self.validation_period_days = validation_period_days
        self.games = None
        self.train_end_date = None
        self.val_end_date = None
        
    def _load_and_split_data(self):
        """Load data and create train/validation split"""
        # Load games data
        self.games = pd.read_csv("./data/games.csv", index_col=0, parse_dates=["Date", "Open"])
        
        # Get all game dates
        all_dates = sorted(self.games['Date'].unique())
        
        # Use last validation_period_days for validation
        self.val_end_date = all_dates[-1]
        self.train_end_date = self.val_end_date - timedelta(days=self.validation_period_days)
        
        print(f"Training period: up to {self.train_end_date.date()}")
        print(f"Validation period: {self.train_end_date.date()} to {self.val_end_date.date()}")
        
    def evaluate_betting_strategy(self, bet_threshold, bet_size_base, bet_kelly_fraction):
        """
        Evaluate betting strategy on validation period.
        
        Returns:
            Final bankroll (negative for loss, to minimize)
        """
        # Create model with fixed prediction hyperparameters and variable betting parameters
        model = Model(
            # Use optimized prediction hyperparameters (from previous optimization)
            ensamble_required_n=700,
            ensamble_retrain_freq=600,
            pythagorean_power=1.499553707049086,
            pythagorean_regularization=10,
            pythagorean_decay=0.992,
            four_factor_required_n=400,
            four_factor_decay=0.9903001400465671,
            gradient_descent_epochs=12,
            gradient_descent_min_games=800,
            exhaustion_b2b_penalty=1.8708290679206017,
            exhaustion_road_penalty=0.6669690630456186,
            elo_k_base=10,
            elo_home_advantage=50,
            elo_margin_cap=5,
            poisson_regularization=2.6883250153197973,
            poisson_decay=0.992,
            # Betting strategy parameters (being optimized)
            bet_threshold=bet_threshold,
            bet_size_base=bet_size_base,
            bet_kelly_fraction=bet_kelly_fraction
        )
        
        # Initialize environment with games and model
        env = Environment(
            games=self.games.copy(),
            model=model,
            start_date=None,  # Start from beginning
            end_date=self.val_end_date,
            init_bankroll=1000.0,
            min_bet=5,
            max_bet=100
        )
        
        # Run environment
        evaluation = env.run()
        
        # Get final bankroll
        final_bankroll = env.bankroll
        initial_bankroll = 1000.0
        profit = final_bankroll - initial_bankroll
        
        # Return negative profit (since optuna minimizes)
        return -profit
    
    def objective(self, trial):
        """Optuna objective function"""
        
        # Suggest betting strategy parameters
        bet_threshold = trial.suggest_float('bet_threshold', 0.0, 0.15)
        bet_size_base = trial.suggest_float('bet_size_base', 0.005, 0.05)
        bet_kelly_fraction = trial.suggest_float('bet_kelly_fraction', 0.1, 0.5)
        
        try:
            # Evaluate on validation set
            neg_profit = self.evaluate_betting_strategy(
                bet_threshold=bet_threshold,
                bet_size_base=bet_size_base,
                bet_kelly_fraction=bet_kelly_fraction
            )
            
            return neg_profit
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def optimize(self, n_trials=50, n_jobs=1, storage=None, study_name='betting_strategy_optimization'):
        """
        Run optimization
        
        Args:
            n_trials: Number of trials to run
            n_jobs: Number of parallel workers (1 for sequential, -1 for all CPUs)
            storage: Database URL for parallel execution (auto-generated if None)
            study_name: Name of the optimization study
            
        Returns:
            Dictionary with best parameters
        """
        # Load and split data
        self._load_and_split_data()
        
        # Setup storage for parallel execution
        if n_jobs != 1 and storage is None:
            storage = f'sqlite:///{study_name}.db'
            print(f"Using SQLite storage: {storage}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        
        # Run optimization
        print(f"\nStarting optimization with {n_trials} trials...")
        print(f"Parallel workers: {n_jobs}")
        print(f"Objective: Maximize final bankroll\n")
        
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        
        print(f"\nBest trial:")
        print(f"  Profit: ${-study.best_value:.2f}")
        print(f"  Params: {study.best_params}")
        
        # Calculate metrics for best trial
        best_params = study.best_params
        print(f"\nBest Parameters:")
        print(f"  bet_threshold: {best_params['bet_threshold']:.4f}")
        print(f"  bet_size_base: {best_params['bet_size_base']:.4f}")
        print(f"  bet_kelly_fraction: {best_params['bet_kelly_fraction']:.4f}")
        
        return study.best_params


def main():
    parser = argparse.ArgumentParser(description='Optimize betting strategy hyperparameters')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel workers: 1 (sequential), 4, -1 (all CPUs) (default: 1)')
    parser.add_argument('--study-name', type=str, default='betting_strategy_optimization',
                       help='Name of the optimization study (default: betting_strategy_optimization)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for storage (default: auto-generated for parallel)')
    parser.add_argument('--validation-period', type=int, default=365,
                       help='Validation period in days (default: 365)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BettingOptimizer(validation_period_days=args.validation_period)
    
    # Run optimization
    best_params = optimizer.optimize(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name
    )
    
    print("\n" + "="*60)
    print("To use these parameters in your model:")
    print("="*60)
    print(f"""
model = Model(
    bet_threshold={best_params['bet_threshold']},
    bet_size_base={best_params['bet_size_base']},
    bet_kelly_fraction={best_params['bet_kelly_fraction']}
)
""")


if __name__ == '__main__':
    main()
