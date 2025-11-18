"""
Optuna-based optimization for hybrid betting strategies
Finds optimal parameters for combined Value-Kelly approaches
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout

from environment import Environment
from hybrid_betting_strategies import (
    HybridValueKellyStrategy,
    AdaptiveHybridStrategy,
    EnsembleStrategy,
)


class HybridStrategyOptimizer:
    """
    Optimize hybrid betting strategies using Bayesian optimization
    """
    
    def __init__(
        self,
        games_data: pd.DataFrame,
        strategy_type: str = "hybrid_value_kelly",
        n_trials: int = 50,
        train_split: float = 0.8,
        initial_bankroll: float = 1000.0,
        silent: bool = True,
    ):
        """
        Args:
            games_data: DataFrame with historical games
            strategy_type: One of ['hybrid_value_kelly', 'adaptive_hybrid', 'ensemble']
            n_trials: Number of optimization trials
            train_split: Fraction of data for training (rest for testing)
            initial_bankroll: Starting bankroll for simulation
            silent: If True, suppress verbose output
        """
        self.games_data = games_data.copy()
        self.strategy_type = strategy_type
        self.n_trials = n_trials
        self.train_split = train_split
        self.initial_bankroll = initial_bankroll
        self.silent = silent
        
        # Split data
        self._split_data()
        
        # Store results
        self.study = None
        self.best_params = None
        self.best_train_bankroll = None
        
    def _split_data(self):
        """Split data into train and test sets with proper temporal separation"""
        self.games_data["Date"] = pd.to_datetime(self.games_data["Date"])
        self.games_data = self.games_data.sort_values("Date")
        
        # Calculate split point
        n_games = len(self.games_data)
        split_idx = int(n_games * self.train_split)
        
        # Get split date
        train_end_date = self.games_data.iloc[split_idx - 1]["Date"]
        test_start_date = train_end_date + pd.Timedelta(days=1)  # 1-day gap
        
        # Split data
        self.train_games = self.games_data[
            self.games_data["Date"] <= train_end_date
        ].copy()
        self.test_games = self.games_data[
            self.games_data["Date"] >= test_start_date
        ].copy()
        
        print(f"Training games: {len(self.train_games)}")
        print(f"Testing games: {len(self.test_games)}")
        print(f"Train end date: {train_end_date.date()}")
        print(f"Test start date: {test_start_date.date()}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        
        # Sample hyperparameters based on strategy type
        if self.strategy_type == "hybrid_value_kelly":
            params = {
                "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
                "home_advantage": trial.suggest_float("home_advantage", 0, 200),
                "rolling_window": trial.suggest_int("rolling_window", 3, 20),
                "min_edge": trial.suggest_float("min_edge", 0.04, 0.15),
                "kelly_fraction": trial.suggest_float("kelly_fraction", 0.05, 0.30),
                "value_multiplier": trial.suggest_float("value_multiplier", 1.0, 3.0),
                "confidence_threshold": trial.suggest_float("confidence_threshold", 0.50, 0.75),
                "form_adjustment_weight": trial.suggest_float("form_adjustment_weight", 0, 0.3),
                "streak_adjustment_weight": trial.suggest_float("streak_adjustment_weight", 0, 0.2),
                "initial_rating": trial.suggest_float("initial_rating", 1200, 1800),
            }
            model_class = HybridValueKellyStrategy
            
        elif self.strategy_type == "adaptive_hybrid":
            params = {
                "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
                "home_advantage": trial.suggest_float("home_advantage", 0, 200),
                "rolling_window": trial.suggest_int("rolling_window", 3, 20),
                "aggressive_min_edge": trial.suggest_float("aggressive_min_edge", 0.03, 0.10),
                "conservative_min_edge": trial.suggest_float("conservative_min_edge", 0.08, 0.18),
                "aggressive_kelly": trial.suggest_float("aggressive_kelly", 0.15, 0.35),
                "conservative_kelly": trial.suggest_float("conservative_kelly", 0.05, 0.15),
                "switch_threshold": trial.suggest_float("switch_threshold", 0.80, 0.95),
                "form_adjustment_weight": trial.suggest_float("form_adjustment_weight", 0, 0.3),
                "streak_adjustment_weight": trial.suggest_float("streak_adjustment_weight", 0, 0.2),
                "initial_rating": trial.suggest_float("initial_rating", 1200, 1800),
            }
            model_class = AdaptiveHybridStrategy
            
        elif self.strategy_type == "ensemble":
            params = {
                "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
                "home_advantage": trial.suggest_float("home_advantage", 0, 200),
                "rolling_window": trial.suggest_int("rolling_window", 3, 20),
                "min_edge": trial.suggest_float("min_edge", 0.04, 0.12),
                "kelly_fraction": trial.suggest_float("kelly_fraction", 0.10, 0.25),
                "form_adjustment_weight": trial.suggest_float("form_adjustment_weight", 0, 0.3),
                "streak_adjustment_weight": trial.suggest_float("streak_adjustment_weight", 0, 0.2),
                "initial_rating": trial.suggest_float("initial_rating", 1200, 1800),
            }
            model_class = EnsembleStrategy
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Create model
        model = model_class(**params)
        
        # Run simulation on training data
        env = Environment(
            model=model,
            games=self.train_games,
            init_bankroll=self.initial_bankroll,
        )
        
        if self.silent:
            # Suppress output
            f = StringIO()
            with redirect_stdout(f):
                env.run()
        else:
            env.run()
        
        final_bankroll = env.bankroll
        
        # Pruning: stop if bankroll drops too low
        if final_bankroll < self.initial_bankroll * 0.1:
            raise optuna.TrialPruned()
        
        return final_bankroll
    
    def optimize(self) -> dict:
        """Run optimization"""
        print(f"\nOptimizing {self.strategy_type} strategy...")
        print(f"Trials: {self.n_trials}")
        print("-" * 60)
        
        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_train_bankroll = self.study.best_value
        
        print(f"\nOptimization complete!")
        print(f"Best training bankroll: ${self.best_train_bankroll:.2f}")
        print(f"\nBest parameters:")
        for key, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")
        
        return self.best_params
    
    def evaluate_on_test(self) -> dict:
        """Evaluate best model on test set"""
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        print(f"\nEvaluating on test set...")
        print("-" * 60)
        
        # Create model with best parameters
        if self.strategy_type == "hybrid_value_kelly":
            model = HybridValueKellyStrategy(**self.best_params)
        elif self.strategy_type == "adaptive_hybrid":
            model = AdaptiveHybridStrategy(**self.best_params)
        elif self.strategy_type == "ensemble":
            model = EnsembleStrategy(**self.best_params)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Run on test set
        env = Environment(
            model=model,
            games=self.test_games,
            init_bankroll=self.initial_bankroll,
        )
        
        if self.silent:
            f = StringIO()
            with redirect_stdout(f):
                env.run()
        else:
            env.run()
        
        # Calculate metrics
        final_bankroll = env.bankroll
        profit = final_bankroll - self.initial_bankroll
        roi = (profit / self.initial_bankroll) * 100
        
        # Calculate max drawdown
        history = env.get_history()
        if len(history) > 0:
            cummax = history["Bankroll"].cummax()
            drawdown = ((cummax - history["Bankroll"]) / cummax) * 100
            max_drawdown = drawdown.max() if not drawdown.empty else 0
        else:
            max_drawdown = 0
        
        metrics = {
            "final_bankroll": final_bankroll,
            "profit": profit,
            "roi": roi,
            "max_drawdown": max_drawdown,
            "n_games": len(self.test_games),
        }
        
        print(f"\nTest Results:")
        print(f"  Final Bankroll: ${final_bankroll:.2f}")
        print(f"  Profit/Loss:    ${profit:+.2f}")
        print(f"  ROI:            {roi:+.2f}%")
        print(f"  Max Drawdown:   {max_drawdown:.2f}%")
        
        return metrics
    
    def save_results(self, filename: str = None):
        """Save optimization results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_{self.strategy_type}_optimization_{timestamp}.json"
        
        results = {
            "strategy_type": self.strategy_type,
            "n_trials": self.n_trials,
            "best_params": self.best_params,
            "best_train_bankroll": self.best_train_bankroll,
            "train_games": len(self.train_games),
            "test_games": len(self.test_games),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        return filename


def compare_hybrid_strategies(
    games_data: pd.DataFrame,
    n_trials: int = 30,
    initial_bankroll: float = 1000.0,
) -> pd.DataFrame:
    """
    Compare all hybrid strategies
    
    Returns:
        DataFrame with comparison results
    """
    
    strategies = ["hybrid_value_kelly", "adaptive_hybrid", "ensemble"]
    results = []
    
    for strategy_type in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing: {strategy_type.upper()}")
        print(f"{'=' * 60}")
        
        optimizer = HybridStrategyOptimizer(
            games_data=games_data,
            strategy_type=strategy_type,
            n_trials=n_trials,
            initial_bankroll=initial_bankroll,
            silent=True,
        )
        
        # Optimize
        best_params = optimizer.optimize()
        
        # Evaluate
        test_metrics = optimizer.evaluate_on_test()
        
        # Save results
        optimizer.save_results()
        
        # Store for comparison
        results.append({
            "Strategy": strategy_type,
            "Final Bankroll": test_metrics["final_bankroll"],
            "ROI (%)": test_metrics["roi"],
            "Max Drawdown (%)": test_metrics["max_drawdown"],
            "Best Train Bankroll": optimizer.best_train_bankroll,
        })
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("Final Bankroll", ascending=False)
    
    print(f"\n{'=' * 60}")
    print("HYBRID STRATEGIES COMPARISON")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_strategies_comparison_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nComparison saved to: {filename}")
    
    return df


if __name__ == "__main__":
    # Load data
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hybrid betting strategies")
    parser.add_argument(
        "--strategy",
        choices=["hybrid_value_kelly", "adaptive_hybrid", "ensemble"],
        default="hybrid_value_kelly",
        help="Strategy to optimize",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all hybrid strategies",
    )
    parser.add_argument(
        "--data",
        default="data/games.csv",
        help="Path to games data CSV",
    )
    
    args = parser.parse_args()
    
    # Load games data
    print(f"Loading data from: {args.data}")
    games_data = pd.read_csv(args.data, parse_dates=['Date', 'Open'])
    print(f"Loaded {len(games_data)} games")
    
    if args.all:
        # Compare all strategies
        compare_hybrid_strategies(games_data, n_trials=args.trials)
    else:
        # Optimize single strategy
        optimizer = HybridStrategyOptimizer(
            games_data=games_data,
            strategy_type=args.strategy,
            n_trials=args.trials,
            silent=True,
        )
        
        best_params = optimizer.optimize()
        test_metrics = optimizer.evaluate_on_test()
        optimizer.save_results()
