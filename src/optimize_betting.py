"""
Betting Strategy Hyperparameter Optimization

This script tunes betting strategy parameters to maximize final bankroll while keeping 
prediction model hyperparameters fixed.

Optimized parameters:
    - bet_threshold: Minimum edge required to place bet
    - bet_size_base: Base bet size as fraction of bankroll
    - bet_kelly_fraction: Kelly criterion fraction
    - bet_min_confidence: Minimum probability to consider betting
    - bet_max_exposure: Maximum total exposure per game
    - bet_size_max_multiplier: Maximum multiplier on base bet size
    - use_overround_adjustment: Whether to normalize implied probabilities
    - confidence_scaling: Whether to scale bets by prediction confidence

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
    
    def __init__(self, validation_period_days=365, data_path="../data/games.csv"):
        """
        Args:
            validation_period_days: Number of days to use for validation
            data_path: Path to the games CSV file
        """
        self.validation_period_days = validation_period_days
        self.data_path = data_path
        
    def _load_and_split_data(self):
        """Load data and create train/validation split - called in each trial for parallel safety"""
        # Load games data (each worker loads its own copy)
        games = pd.read_csv(self.data_path, index_col=0, parse_dates=["Date", "Open"])
        
        # Get all game dates
        all_dates = sorted(games['Date'].unique())
        
        # Use last validation_period_days for validation
        val_end_date = all_dates[-1]
        train_end_date = val_end_date - timedelta(days=self.validation_period_days)
        
        return games, train_end_date, val_end_date
        
    def evaluate_betting_strategy(self, games, val_end_date, bet_threshold, bet_size_base, bet_kelly_fraction,
                                  bet_min_confidence, bet_max_exposure, bet_size_max_multiplier,
                                  use_overround_adjustment, confidence_scaling):
        """
        Evaluate betting strategy on validation period.
        
        Args:
            games: DataFrame with game data
            val_end_date: End date for validation period
            [betting parameters]
        
        Returns:
            Final bankroll (negative for loss, to minimize)
        """
        # Create model with fixed prediction hyperparameters and variable betting parameters
        model = Model(
            # Use optimized prediction hyperparameters (from 3-way multinomial optimization)
            ensamble_required_n=700,
            ensamble_retrain_freq=300,
            pythagorean_power=0.9815160790897732,
            pythagorean_regularization=10,
            pythagorean_decay=0.992,
            four_factor_required_n=400,
            four_factor_decay=0.9901321052235277,
            gradient_descent_epochs=12,
            gradient_descent_min_games=800,
            exhaustion_b2b_penalty=1.7896362263470917,
            exhaustion_road_penalty=0.6151810640568766,
            elo_k_base=20,
            elo_home_advantage=45,
            elo_margin_cap=5,
            poisson_regularization=3.319620279992762,
            poisson_decay=0.992,
            # Rolling window parameters (optimized defaults)
            rolling_window_short=5,
            rolling_window_med=10,
            rolling_window_long=25,
            # Betting strategy parameters (being optimized)
            bet_threshold=bet_threshold,
            bet_size_base=bet_size_base,
            bet_kelly_fraction=bet_kelly_fraction,
            bet_min_confidence=bet_min_confidence,
            bet_max_exposure=bet_max_exposure,
            bet_size_max_multiplier=bet_size_max_multiplier,
            use_overround_adjustment=use_overround_adjustment,
            confidence_scaling=confidence_scaling
        )
        
        # Initialize environment with games and model
        env = Environment(
            games=games.copy(),
            model=model,
            start_date=None,  # Start from beginning
            end_date=val_end_date,
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
        
        # Return profit (optuna will maximize this)
        return profit
    
    def objective(self, trial):
        """Optuna objective function"""
        
        # Load data for this trial (each worker loads its own copy - no contention!)
        games, train_end_date, val_end_date = self._load_and_split_data()
        
        # Suggest betting strategy parameters
        # Edge threshold: 0-12% (adjusted from 0-15% to be more realistic)
        bet_threshold = trial.suggest_float('bet_threshold', 0.0, 0.12)
        
        # Base bet size: 0.5-6% of bankroll (adjusted from 0.5-5%)
        bet_size_base = trial.suggest_float('bet_size_base', 0.005, 0.06)
        
        # Kelly fraction: 10-50% (unchanged, standard range)
        bet_kelly_fraction = trial.suggest_float('bet_kelly_fraction', 0.1, 0.5)
        
        # Minimum confidence: 30-60% (new parameter)
        bet_min_confidence = trial.suggest_float('bet_min_confidence', 0.30, 0.60)
        
        # Max exposure per game: 8-25% of bankroll (new parameter)
        bet_max_exposure = trial.suggest_float('bet_max_exposure', 0.08, 0.25)
        
        # Max bet size multiplier: 1.5-5.0x base size (new parameter)
        bet_size_max_multiplier = trial.suggest_float('bet_size_max_multiplier', 1.5, 5.0)
        
        # Overround adjustment: categorical choice (new parameter)
        use_overround_adjustment = trial.suggest_categorical('use_overround_adjustment', [True, False])
        
        # Confidence scaling: categorical choice (new parameter)
        confidence_scaling = trial.suggest_categorical('confidence_scaling', [True, False])
        
        try:
            # Evaluate on validation set (pass games data to avoid shared state)
            profit = self.evaluate_betting_strategy(
                games=games,
                val_end_date=val_end_date,
                bet_threshold=bet_threshold,
                bet_size_base=bet_size_base,
                bet_kelly_fraction=bet_kelly_fraction,
                bet_min_confidence=bet_min_confidence,
                bet_max_exposure=bet_max_exposure,
                bet_size_max_multiplier=bet_size_max_multiplier,
                use_overround_adjustment=use_overround_adjustment,
                confidence_scaling=confidence_scaling
            )
            
            return profit
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')  # Return worst possible value for maximization
    
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
        # Load data once to print info (each worker will load its own copy)
        games, train_end_date, val_end_date = self._load_and_split_data()
        print(f"Training period: up to {train_end_date.date()}")
        print(f"Validation period: {train_end_date.date()} to {val_end_date.date()}")
        
        # Setup storage for parallel execution
        if n_jobs != 1 and storage is None:
            storage = f'sqlite:///{study_name}.db'
            print(f"Using SQLite storage: {storage}")
            print(f"Note: Each worker loads its own data copy for parallel safety")
        
        # Create study with optimized SQLite settings for parallel execution
        if storage and storage.startswith('sqlite'):
            # Use WAL mode for better concurrent writes
            import sqlite3
            db_path = storage.replace('sqlite:///', '')
            conn = sqlite3.connect(db_path)
            conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for concurrency
            conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
            conn.close()
        
        study = optuna.create_study(
            direction='maximize',  # Maximize profit
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
        print(f"  Profit: ${study.best_value:.2f}")
        print(f"  Params: {study.best_params}")
        
        # Calculate metrics for best trial
        best_params = study.best_params
        print(f"\nBest Parameters:")
        print(f"  bet_threshold: {best_params['bet_threshold']:.4f}")
        print(f"  bet_size_base: {best_params['bet_size_base']:.4f}")
        print(f"  bet_kelly_fraction: {best_params['bet_kelly_fraction']:.4f}")
        print(f"  bet_min_confidence: {best_params['bet_min_confidence']:.4f}")
        print(f"  bet_max_exposure: {best_params['bet_max_exposure']:.4f}")
        print(f"  bet_size_max_multiplier: {best_params['bet_size_max_multiplier']:.4f}")
        print(f"  use_overround_adjustment: {best_params['use_overround_adjustment']}")
        print(f"  confidence_scaling: {best_params['confidence_scaling']}")
        
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
    bet_kelly_fraction={best_params['bet_kelly_fraction']},
    bet_min_confidence={best_params['bet_min_confidence']},
    bet_max_exposure={best_params['bet_max_exposure']},
    bet_size_max_multiplier={best_params['bet_size_max_multiplier']},
    use_overround_adjustment={best_params['use_overround_adjustment']},
    confidence_scaling={best_params['confidence_scaling']}
)
""")


if __name__ == '__main__':
    main()
