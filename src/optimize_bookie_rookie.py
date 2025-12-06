"""
Bookie Rookie Model Comprehensive Hyperparameter Optimization

This script optimizes BOTH:
1. Prediction model parameters (ensemble, pythagorean, four_factor, elo, etc.)
2. Betting strategy parameters (min_edge, kelly_fraction, confidence weights, etc.)

The goal is to find the optimal combination that maximizes profit on the validation set.

Usage:
    cd src && uv run optimize_bookie_rookie.py --n-trials 100 --n-jobs 4
    cd src && uv run optimize_bookie_rookie.py --n-trials 50  # Sequential
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import optuna
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_bookie_rookie import Model
from environment import Environment


class BookieRookieOptimizer:
    """Optimize both prediction and betting parameters for bookie rookie model"""
    
    # def __init__(self, validation_period_days=365, data_path="../data/bookie_rookie_games.csv"):
    def __init__(self, validation_period_days=365, data_path="../data/train_games.csv"):
        """
        Args:
            validation_period_days: Number of days to use for validation
            data_path: Path to the games CSV file
        """
        self.validation_period_days = validation_period_days
        self.data_path = data_path
        
    def _load_and_split_data(self):
        """Load data and create train/validation split - called in each trial for parallel safety"""
        # Load bookie rookie games data (each worker loads its own copy)
        games = pd.read_csv(self.data_path, index_col=0, parse_dates=["Date", "Open"])
        
        # Get all game dates
        all_dates = sorted(games['Date'].unique())
        
        # Use last validation_period_days for validation
        val_end_date = all_dates[-1]
        train_end_date = val_end_date - timedelta(days=self.validation_period_days)
        
        return games, train_end_date, val_end_date
        
    def evaluate_strategy(self, games, val_end_date, **params):
        """
        Evaluate model with given parameters on validation period.
        
        Args:
            games: DataFrame with game data
            val_end_date: End date for validation period
            **params: Model parameters
        
        Returns:
            Final profit (to maximize)
        """
        # Create model with all parameters
        model = Model(**params)
        
        # Initialize environment with games and model
        env = Environment(
            games=games.copy(),
            model=model,
            start_date=None,  # Start from beginning
            end_date=val_end_date,
            init_bankroll=1000.0,
            min_bet=1,
            max_bet=250
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
        """Optuna objective function - suggests and evaluates parameters"""
        
        # Load data for this trial (each worker loads its own copy - no contention!)
        games, train_end_date, val_end_date = self._load_and_split_data()
        
        # ===== PREDICTION MODEL PARAMETERS =====
        
        # Ensemble parameters
        ensamble_required_n = trial.suggest_int('ensamble_required_n', 300, 800, step=100)
        ensamble_retrain_freq = trial.suggest_int('ensamble_retrain_freq', 150, 400, step=50)
        
        # Pythagorean model
        pythagorean_power = trial.suggest_float('pythagorean_power', 0.8, 1.5)
        pythagorean_regularization = trial.suggest_int('pythagorean_regularization', 5, 30, step=5)
        pythagorean_decay = trial.suggest_float('pythagorean_decay', 0.985, 0.998)
        
        # Four Factor model
        four_factor_required_n = trial.suggest_int('four_factor_required_n', 200, 600, step=100)
        four_factor_decay = trial.suggest_float('four_factor_decay', 0.990, 0.998)
        
        # Gradient Descent
        gradient_descent_epochs = trial.suggest_int('gradient_descent_epochs', 5, 20)
        gradient_descent_min_games = trial.suggest_int('gradient_descent_min_games', 400, 800, step=100)
        
        # Exhaustion model
        exhaustion_b2b_penalty = trial.suggest_float('exhaustion_b2b_penalty', 1.0, 2.5)
        exhaustion_road_penalty = trial.suggest_float('exhaustion_road_penalty', 0.3, 0.8)
        
        # ELO system
        elo_k_base = trial.suggest_int('elo_k_base', 10, 30, step=5)
        elo_home_advantage = trial.suggest_int('elo_home_advantage', 20, 60, step=5)
        elo_margin_cap = trial.suggest_int('elo_margin_cap', 2, 7)
        
        # Poisson model
        poisson_regularization = trial.suggest_float('poisson_regularization', 1.5, 5.0)
        poisson_decay = trial.suggest_float('poisson_decay', 0.985, 0.998)
        
        # Rolling window features
        rolling_window_short = trial.suggest_int('rolling_window_short', 3, 8)
        rolling_window_med = trial.suggest_int('rolling_window_med', 8, 15)
        rolling_window_long = trial.suggest_int('rolling_window_long', 16, 30)
        
        # ===== BETTING STRATEGY PARAMETERS =====
        
        # Minimum edge: Lower = more bets, higher = more selective
        # Range: 3-8% (reasonable profitable edge threshold)
        pepe_min_edge = trial.suggest_float('pepe_min_edge', 0.03, 0.08)
        
        # Kelly fraction: Lower = conservative, higher = aggressive
        # Range: 30-70% (standard Kelly is often too aggressive)
        pepe_kelly_fraction = trial.suggest_float('pepe_kelly_fraction', 0.30, 0.70)
        
        # Value multiplier: Cap on how much to increase bet based on value
        # Range: 1.5-4.0x (limits bet size on high-value opportunities)
        pepe_value_multiplier = trial.suggest_float('pepe_value_multiplier', 1.5, 4.0)
        
        # Confidence threshold: Minimum confidence to place any bet
        # Range: 45-70% (balance between selectivity and opportunity)
        pepe_confidence_threshold = trial.suggest_float('pepe_confidence_threshold', 0.45, 0.70)
        
        # Max bet fraction: Largest single bet as % of bankroll
        # Range: 10-30% (risk management - never bet too much on one game)
        pepe_max_bet_fraction = trial.suggest_float('pepe_max_bet_fraction', 0.10, 0.30)
        
        # Max total exposure: Total capital at risk across all pending bets
        # Range: 50-90% (balance between opportunity and safety)
        pepe_max_total_exposure = trial.suggest_float('pepe_max_total_exposure', 0.50, 0.90)
        
        # Confidence weights: How to calculate confidence score
        # Sample only 2 weights, derive the 3rd to ensure they sum to 1.0
        pepe_confidence_weight_separation = trial.suggest_float('pepe_confidence_weight_separation', 0.1, 0.8)
        pepe_confidence_weight_certainty = trial.suggest_float('pepe_confidence_weight_certainty', 0.1, 0.8)
        pepe_confidence_weight_form = 1.0 - pepe_confidence_weight_separation - pepe_confidence_weight_certainty
        
        # Ensure form weight is valid
        if pepe_confidence_weight_form < 0.05 or pepe_confidence_weight_form > 0.8:
            return float('-inf')
        
        try:
            # Evaluate on validation set (pass games data to avoid shared state)
            profit = self.evaluate_strategy(
                games=games,
                val_end_date=val_end_date,
                # Prediction parameters
                ensamble_required_n=ensamble_required_n,
                ensamble_retrain_freq=ensamble_retrain_freq,
                pythagorean_power=pythagorean_power,
                pythagorean_regularization=pythagorean_regularization,
                pythagorean_decay=pythagorean_decay,
                four_factor_required_n=four_factor_required_n,
                four_factor_decay=four_factor_decay,
                gradient_descent_epochs=gradient_descent_epochs,
                gradient_descent_min_games=gradient_descent_min_games,
                exhaustion_b2b_penalty=exhaustion_b2b_penalty,
                exhaustion_road_penalty=exhaustion_road_penalty,
                elo_k_base=elo_k_base,
                elo_home_advantage=elo_home_advantage,
                elo_margin_cap=elo_margin_cap,
                poisson_regularization=poisson_regularization,
                poisson_decay=poisson_decay,
                rolling_window_short=rolling_window_short,
                rolling_window_med=rolling_window_med,
                rolling_window_long=rolling_window_long,
                # Betting parameters
                pepe_min_edge=pepe_min_edge,
                pepe_kelly_fraction=pepe_kelly_fraction,
                pepe_value_multiplier=pepe_value_multiplier,
                pepe_confidence_threshold=pepe_confidence_threshold,
                pepe_max_bet_fraction=pepe_max_bet_fraction,
                pepe_max_total_exposure=pepe_max_total_exposure,
                pepe_confidence_weight_separation=pepe_confidence_weight_separation,
                pepe_confidence_weight_certainty=pepe_confidence_weight_certainty,
                pepe_confidence_weight_form=pepe_confidence_weight_form
            )
            
            return profit
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')
    
    def optimize(self, n_trials=50, n_jobs=1, storage=None, study_name='bookie_rookie_optimization'):
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
        print(f"Total games: {len(games)}")
        
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
        print(f"Objective: Maximize final profit\n")
        
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        
        print(f"\nBest trial:")
        print(f"  Profit: ${study.best_value:.2f}")
        print(f"  ROI: {(study.best_value / 1000.0) * 100:.2f}%")
        
        # Calculate derived form weight
        best_params = study.best_params
        pepe_confidence_weight_form = 1.0 - best_params['pepe_confidence_weight_separation'] - best_params['pepe_confidence_weight_certainty']
        
        print(f"\nBest Parameters:")
        print("\n--- Prediction Model ---")
        print(f"  ensamble_required_n: {best_params['ensamble_required_n']}")
        print(f"  ensamble_retrain_freq: {best_params['ensamble_retrain_freq']}")
        print(f"  pythagorean_power: {best_params['pythagorean_power']:.4f}")
        print(f"  pythagorean_regularization: {best_params['pythagorean_regularization']}")
        print(f"  pythagorean_decay: {best_params['pythagorean_decay']:.4f}")
        print(f"  four_factor_required_n: {best_params['four_factor_required_n']}")
        print(f"  four_factor_decay: {best_params['four_factor_decay']:.4f}")
        print(f"  gradient_descent_epochs: {best_params['gradient_descent_epochs']}")
        print(f"  gradient_descent_min_games: {best_params['gradient_descent_min_games']}")
        print(f"  exhaustion_b2b_penalty: {best_params['exhaustion_b2b_penalty']:.4f}")
        print(f"  exhaustion_road_penalty: {best_params['exhaustion_road_penalty']:.4f}")
        print(f"  elo_k_base: {best_params['elo_k_base']}")
        print(f"  elo_home_advantage: {best_params['elo_home_advantage']}")
        print(f"  elo_margin_cap: {best_params['elo_margin_cap']}")
        print(f"  poisson_regularization: {best_params['poisson_regularization']:.4f}")
        print(f"  poisson_decay: {best_params['poisson_decay']:.4f}")
        print(f"  rolling_window_short: {best_params['rolling_window_short']}")
        print(f"  rolling_window_med: {best_params['rolling_window_med']}")
        print(f"  rolling_window_long: {best_params['rolling_window_long']}")
        
        print("\n--- Betting Strategy ---")
        print(f"  pepe_min_edge: {best_params['pepe_min_edge']:.4f}")
        print(f"  pepe_kelly_fraction: {best_params['pepe_kelly_fraction']:.4f}")
        print(f"  pepe_value_multiplier: {best_params['pepe_value_multiplier']:.4f}")
        print(f"  pepe_confidence_threshold: {best_params['pepe_confidence_threshold']:.4f}")
        print(f"  pepe_max_bet_fraction: {best_params['pepe_max_bet_fraction']:.4f}")
        print(f"  pepe_max_total_exposure: {best_params['pepe_max_total_exposure']:.4f}")
        print(f"  pepe_confidence_weight_separation: {best_params['pepe_confidence_weight_separation']:.4f}")
        print(f"  pepe_confidence_weight_certainty: {best_params['pepe_confidence_weight_certainty']:.4f}")
        print(f"  pepe_confidence_weight_form: {pepe_confidence_weight_form:.4f}")
        
        # Save results to JSON
        results = {
            'best_profit': study.best_value,
            'best_roi': (study.best_value / 1000.0) * 100,
            'best_params': best_params,
            'pepe_confidence_weight_form': pepe_confidence_weight_form,
            'n_trials': n_trials,
            'validation_period_days': self.validation_period_days
        }
        
        with open('bookie_rookie_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to bookie_rookie_optimization_results.json")
        
        return best_params


def main():
    parser = argparse.ArgumentParser(description='Optimize bookie rookie model (prediction + betting parameters)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel workers: 1 (sequential), 4, -1 (all CPUs) (default: 1)')
    parser.add_argument('--study-name', type=str, default='bookie_rookie_optimization',
                       help='Name of the optimization study (default: bookie_rookie_optimization)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for storage (default: auto-generated for parallel)')
    parser.add_argument('--validation-period', type=int, default=365,
                       help='Validation period in days (default: 365)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BookieRookieOptimizer(validation_period_days=args.validation_period)
    
    # Run optimization
    best_params = optimizer.optimize(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name
    )
    
    print("\n" + "="*60)
    print("To use these parameters in your model, update model_bookie_rookie.py")
    print("with the values shown above.")
    print("="*60)


if __name__ == '__main__':
    main()
