"""
Optuna-based optimization for contrarian betting strategies
Optimizes thresholds and bias multipliers for maximum bankroll
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from datetime import datetime
from typing import Dict, Any
import io
from contextlib import redirect_stdout

from contrarian_betting_strategies import (
    ContrarianBettingStrategy,
    AdaptiveContrarianStrategy,
)
from environment import Environment


class ContrarianStrategyOptimizer:
    """
    Optimize contrarian betting strategies using Optuna
    """
    
    def __init__(
        self,
        games_data: pd.DataFrame,
        strategy_type: str = "contrarian",  # "contrarian" or "adaptive_contrarian"
        n_trials: int = 50,
        train_ratio: float = 0.8,
        silent: bool = True,
    ):
        self.games_data = games_data
        self.strategy_type = strategy_type
        self.n_trials = n_trials
        self.train_ratio = train_ratio
        self.silent = silent
        
        # Split data
        self._split_data()
        
        # Storage for results
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def _split_data(self):
        """Split data into train and test sets with proper temporal separation"""
        # Sort by date
        if "Date" in self.games_data.columns:
            self.games_data["Date"] = pd.to_datetime(self.games_data["Date"])
            self.games_data = self.games_data.sort_values("Date")
        
        # Calculate split point
        n_games = len(self.games_data)
        train_size = int(n_games * self.train_ratio)
        
        # Get split date
        train_end_date = self.games_data.iloc[train_size - 1]["Date"]
        # Add 1 day gap to prevent data leakage
        test_start_date = train_end_date + pd.Timedelta(days=1)
        
        # Split
        self.train_data = self.games_data[self.games_data["Date"] <= train_end_date]
        self.test_data = self.games_data[self.games_data["Date"] >= test_start_date]
        
        print(f"Training set: {len(self.train_data)} games")
        print(f"Test set: {len(self.test_data)} games")
        print(f"Train end: {train_end_date.date()}")
        print(f"Test start: {test_start_date.date()}")
    
    def _objective_contrarian(self, trial: optuna.Trial) -> float:
        """
        Objective function for ContrarianBettingStrategy
        """
        # Suggest hyperparameters
        params = {
            "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
            "home_advantage": trial.suggest_float("home_advantage", 0, 200),
            "rolling_window": trial.suggest_int("rolling_window", 3, 20),
            "high_confidence_threshold": trial.suggest_float("high_confidence_threshold", 0.60, 0.85),
            "low_confidence_threshold": trial.suggest_float("low_confidence_threshold", 0.25, 0.50),
            "min_edge": trial.suggest_float("min_edge", 0.01, 0.15),
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.05, 0.30),
            "contrarian_kelly_fraction": trial.suggest_float("contrarian_kelly_fraction", 0.05, 0.20),
            "draw_bias_multiplier": trial.suggest_float("draw_bias_multiplier", 1.0, 3.0),
            "underdog_bias_multiplier": trial.suggest_float("underdog_bias_multiplier", 1.0, 2.5),
            "form_adjustment_weight": trial.suggest_float("form_adjustment_weight", 0, 0.3),
            "streak_adjustment_weight": trial.suggest_float("streak_adjustment_weight", 0, 0.2),
            "initial_rating": trial.suggest_float("initial_rating", 1200, 1800),
        }
        
        # Ensure high threshold > low threshold
        if params["high_confidence_threshold"] <= params["low_confidence_threshold"]:
            return -1000  # Invalid configuration
        
        # Create model
        model = ContrarianBettingStrategy(**params)
        
        # Run simulation (silent mode)
        env = Environment(model, self.train_data, initial_bankroll=1000)
        
        if self.silent:
            # Suppress output
            f = io.StringIO()
            with redirect_stdout(f):
                env.run()
        else:
            env.run()
        
        # Get final bankroll
        history = env.get_history()
        final_bankroll = history["Bankroll"].iloc[-1]
        
        # Prune if performance is terrible
        if final_bankroll < 100:  # Lost 90%+
            raise optuna.TrialPruned()
        
        return final_bankroll
    
    def _objective_adaptive_contrarian(self, trial: optuna.Trial) -> float:
        """
        Objective function for AdaptiveContrarianStrategy
        """
        # Suggest hyperparameters
        params = {
            "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
            "home_advantage": trial.suggest_float("home_advantage", 0, 200),
            "rolling_window": trial.suggest_int("rolling_window", 3, 20),
            "threshold_very_high": trial.suggest_float("threshold_very_high", 0.70, 0.90),
            "threshold_high": trial.suggest_float("threshold_high", 0.55, 0.75),
            "threshold_medium_low": trial.suggest_float("threshold_medium_low", 0.35, 0.55),
            "threshold_low": trial.suggest_float("threshold_low", 0.20, 0.40),
            "min_edge": trial.suggest_float("min_edge", 0.01, 0.15),
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.05, 0.30),
            "draw_bias": trial.suggest_float("draw_bias", 1.0, 3.0),
            "underdog_bias": trial.suggest_float("underdog_bias", 1.0, 2.5),
            "form_adjustment_weight": trial.suggest_float("form_adjustment_weight", 0, 0.3),
            "streak_adjustment_weight": trial.suggest_float("streak_adjustment_weight", 0, 0.2),
            "initial_rating": trial.suggest_float("initial_rating", 1200, 1800),
        }
        
        # Ensure thresholds are properly ordered
        if not (params["threshold_very_high"] > params["threshold_high"] > 
                params["threshold_medium_low"] > params["threshold_low"]):
            return -1000  # Invalid configuration
        
        # Create model
        model = AdaptiveContrarianStrategy(**params)
        
        # Run simulation
        env = Environment(model, self.train_data, initial_bankroll=1000)
        
        if self.silent:
            f = io.StringIO()
            with redirect_stdout(f):
                env.run()
        else:
            env.run()
        
        # Get final bankroll
        history = env.get_history()
        final_bankroll = history["Bankroll"].iloc[-1]
        
        # Prune if terrible
        if final_bankroll < 100:
            raise optuna.TrialPruned()
        
        return final_bankroll
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization
        """
        print(f"\nOptimizing {self.strategy_type} strategy...")
        print(f"Number of trials: {self.n_trials}")
        print(f"Objective: Maximize final bankroll on training set\n")
        
        # Select objective function
        if self.strategy_type == "contrarian":
            objective_func = self._objective_contrarian
        elif self.strategy_type == "adaptive_contrarian":
            objective_func = self._objective_adaptive_contrarian
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        
        # Optimize
        self.study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"\nOptimization complete!")
        print(f"Best training bankroll: ${self.best_value:.2f}")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value:.4f}")
        
        return self.best_params
    
    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Evaluate best model on test set
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        print(f"\nEvaluating on test set...")
        
        # Create model with best parameters
        if self.strategy_type == "contrarian":
            model = ContrarianBettingStrategy(**self.best_params)
        else:
            model = AdaptiveContrarianStrategy(**self.best_params)
        
        # Run on test set
        env = Environment(model, self.test_data, initial_bankroll=1000)
        
        if self.silent:
            f = io.StringIO()
            with redirect_stdout(f):
                env.run()
        else:
            env.run()
        
        # Calculate metrics
        history = env.get_history()
        final_bankroll = history["Bankroll"].iloc[-1]
        roi = ((final_bankroll - 1000) / 1000) * 100
        
        # Max drawdown (peak-to-trough)
        cummax = history["Bankroll"].cummax()
        drawdown = ((cummax - history["Bankroll"]) / cummax) * 100
        max_drawdown = drawdown.max() if not drawdown.empty else 0
        
        # Volatility and Sharpe
        returns = history["Bankroll"].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        metrics = {
            "final_bankroll": final_bankroll,
            "roi": roi,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
        }
        
        print(f"\nTest Results:")
        print(f"  Final Bankroll: ${final_bankroll:.2f}")
        print(f"  ROI: {roi:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        
        return metrics
    
    def save_results(self, filename: str = None):
        """Save optimization results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contrarian_{self.strategy_type}_optimization_{timestamp}.json"
        
        results = {
            "strategy_type": self.strategy_type,
            "n_trials": self.n_trials,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "train_size": len(self.train_data),
            "test_size": len(self.test_data),
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize contrarian betting strategies")
    parser.add_argument(
        "--strategy",
        type=str,
        default="contrarian",
        choices=["contrarian", "adaptive_contrarian"],
        help="Strategy type to optimize"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/games.csv",
        help="Path to games data"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output during optimization"
    )
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return
    
    games_data = pd.read_csv(args.data, parse_dates=['Date'])
    
    # Run optimization
    optimizer = ContrarianStrategyOptimizer(
        games_data=games_data,
        strategy_type=args.strategy,
        n_trials=args.trials,
        silent=not args.verbose,
    )
    
    best_params = optimizer.optimize()
    test_metrics = optimizer.evaluate_on_test()
    optimizer.save_results()


if __name__ == "__main__":
    main()
