"""
Hybrid Model Betting Strategy Hyperparameter Optimization

This script tunes Pepe's betting strategy parameters while keeping the prediction 
model hyperparameters fixed. The goal is to find the optimal combination of:
- min_edge: Minimum edge required to place bet
- kelly_fraction: Kelly criterion fraction
- value_multiplier: Maximum value multiplier
- confidence_threshold: Minimum confidence to bet
- max_bet_fraction: Maximum bet as fraction of bankroll
- max_total_exposure: Maximum total exposure across all bets

Usage:
    python3 src/optimize_hybrid.py --n-trials 100 --n-jobs 4
    python3 src/optimize_hybrid.py --n-trials 50  # Sequential execution
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

from model_hybrid import Model
from environment import Environment


class HybridOptimizer:
    """Optimize Pepe's betting strategy parameters in hybrid model"""
    
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
        self.games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])
        
        # Get all game dates
        all_dates = sorted(self.games['Date'].unique())
        
        # Use last validation_period_days for validation
        self.val_end_date = all_dates[-1]
        self.train_end_date = self.val_end_date - timedelta(days=self.validation_period_days)
        
        print(f"Training period: up to {self.train_end_date.date()}")
        print(f"Validation period: {self.train_end_date.date()} to {self.val_end_date.date()}")
        
    def evaluate_hybrid_strategy(self, pepe_min_edge, pepe_kelly_fraction, 
                                 pepe_value_multiplier, pepe_confidence_threshold,
                                 pepe_max_bet_fraction, pepe_max_total_exposure):
        """
        Evaluate Pepe's betting strategy parameters on validation period.
        
        Returns:
            Final profit (to maximize)
        """
        # Create hybrid model with fixed prediction hyperparameters and variable betting parameters
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
            rolling_window_short=5,
            rolling_window_med=10,
            rolling_window_long=25,
            # Pepe's betting strategy parameters (being optimized)
            pepe_min_edge=pepe_min_edge,
            pepe_kelly_fraction=pepe_kelly_fraction,
            pepe_value_multiplier=pepe_value_multiplier,
            pepe_confidence_threshold=pepe_confidence_threshold,
            pepe_max_bet_fraction=pepe_max_bet_fraction,
            pepe_max_total_exposure=pepe_max_total_exposure
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
        
        # Return profit (optuna will maximize this)
        return profit
    
    def objective(self, trial):
        """Optuna objective function"""
        
        # Suggest Pepe's betting strategy parameters
        
        # Minimum edge: 2-10% (Pepe's optimized was 4.13%)
        pepe_min_edge = trial.suggest_float('pepe_min_edge', 0.02, 0.10)
        
        # Kelly fraction: 10-50% (Pepe's optimized was 28.6%)
        pepe_kelly_fraction = trial.suggest_float('pepe_kelly_fraction', 0.10, 0.50)
        
        # Value multiplier: 1.5-5.0x (Pepe's optimized was 2.95)
        pepe_value_multiplier = trial.suggest_float('pepe_value_multiplier', 1.5, 5.0)
        
        # Confidence threshold: 40-80% (Pepe's optimized was 59%)
        pepe_confidence_threshold = trial.suggest_float('pepe_confidence_threshold', 0.40, 0.80)
        
        # Max bet fraction: 5-20% of bankroll (Pepe's was 10%)
        pepe_max_bet_fraction = trial.suggest_float('pepe_max_bet_fraction', 0.05, 0.20)
        
        # Max total exposure: 70-100% (Pepe's was 95%)
        pepe_max_total_exposure = trial.suggest_float('pepe_max_total_exposure', 0.70, 1.00)
        
        try:
            # Evaluate on validation set
            profit = self.evaluate_hybrid_strategy(
                pepe_min_edge=pepe_min_edge,
                pepe_kelly_fraction=pepe_kelly_fraction,
                pepe_value_multiplier=pepe_value_multiplier,
                pepe_confidence_threshold=pepe_confidence_threshold,
                pepe_max_bet_fraction=pepe_max_bet_fraction,
                pepe_max_total_exposure=pepe_max_total_exposure
            )
            
            return profit
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')  # Return worst possible value for maximization
    
    def optimize(self, n_trials=50, n_jobs=1, storage=None, study_name='hybrid_strategy_optimization'):
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
        print(f"  pepe_min_edge: {best_params['pepe_min_edge']:.4f}")
        print(f"  pepe_kelly_fraction: {best_params['pepe_kelly_fraction']:.4f}")
        print(f"  pepe_value_multiplier: {best_params['pepe_value_multiplier']:.4f}")
        print(f"  pepe_confidence_threshold: {best_params['pepe_confidence_threshold']:.4f}")
        print(f"  pepe_max_bet_fraction: {best_params['pepe_max_bet_fraction']:.4f}")
        print(f"  pepe_max_total_exposure: {best_params['pepe_max_total_exposure']:.4f}")
        
        return study.best_params


def main():
    parser = argparse.ArgumentParser(description='Optimize hybrid model betting strategy hyperparameters')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel workers: 1 (sequential), 4, -1 (all CPUs) (default: 1)')
    parser.add_argument('--study-name', type=str, default='hybrid_strategy_optimization',
                       help='Name of the optimization study (default: hybrid_strategy_optimization)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for storage (default: auto-generated for parallel)')
    parser.add_argument('--validation-period', type=int, default=365,
                       help='Validation period in days (default: 365)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HybridOptimizer(validation_period_days=args.validation_period)
    
    # Run optimization
    best_params = optimizer.optimize(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name
    )
    
    print("\n" + "="*60)
    print("To use these parameters in your hybrid model:")
    print("="*60)
    print(f"""
from model_hybrid import Model

model = Model(
    pepe_min_edge={best_params['pepe_min_edge']},
    pepe_kelly_fraction={best_params['pepe_kelly_fraction']},
    pepe_value_multiplier={best_params['pepe_value_multiplier']},
    pepe_confidence_threshold={best_params['pepe_confidence_threshold']},
    pepe_max_bet_fraction={best_params['pepe_max_bet_fraction']},
    pepe_max_total_exposure={best_params['pepe_max_total_exposure']}
)
""")


if __name__ == '__main__':
    main()
