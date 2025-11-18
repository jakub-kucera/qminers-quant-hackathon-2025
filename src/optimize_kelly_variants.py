"""
Optuna-based Hyperparameter Optimization for Kelly Criterion Variants

Optimizes Kelly-based betting strategies to find the best parameters for:
- Kelly fraction
- Minimum edge threshold
- Risk management parameters (for Adaptive Kelly)

Uses the same Optuna framework as the main tuner but focuses specifically
on Kelly variant strategies.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Type
import json
from datetime import datetime

from models import EloModel
from betting_strategies_research import (
    FullKellyStrategy,
    AdaptiveKellyStrategy,
)
from environment import Environment


class KellyVariantOptimizer:
    """
    Optimize Kelly Criterion variant strategies using Optuna.
    
    Focuses on finding optimal:
    - Kelly fraction (for fractional Kelly)
    - Minimum edge threshold
    - Elo rating parameters
    - Adaptive Kelly parameters (base_kelly, min_kelly, max_kelly, adaptation_rate)
    """
    
    def __init__(
        self,
        games_data: pd.DataFrame,
        strategy_type: str = "fractional",  # "fractional", "adaptive", "full"
        n_trials: int = 50,
        init_bankroll: float = 1000.0,
        min_bet: float = 5.0,
        max_bet: float = 100.0,
        train_ratio: float = 0.8,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ):
        """
        Initialize Kelly variant optimizer.
        
        Args:
            games_data: DataFrame with game data
            strategy_type: Type of Kelly strategy ("fractional", "adaptive", "full")
            n_trials: Number of optimization trials
            init_bankroll: Initial bankroll
            min_bet: Minimum bet size
            max_bet: Maximum bet size
            train_ratio: Ratio of data for training (rest for validation)
            timeout: Timeout in seconds
            study_name: Name for the Optuna study
        """
        self.games_data = games_data.copy()
        self.strategy_type = strategy_type
        self.n_trials = n_trials
        self.init_bankroll = init_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.train_ratio = train_ratio
        self.timeout = timeout
        
        # Split data
        self._split_data()
        
        # Create study
        study_name = study_name or f"kelly_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        
        self.best_params = None
        self.best_model = None
    
    def _split_data(self):
        """Split data into train and test sets with proper temporal separation."""
        self.games_data = self.games_data.sort_values("Date").reset_index(drop=True)
        
        split_idx = int(len(self.games_data) * self.train_ratio)
        self.train_data = self.games_data.iloc[:split_idx].copy()
        
        # Add 1 day gap to prevent data leakage
        train_end_date = pd.to_datetime(self.train_data["Date"].max())
        test_start_date = train_end_date + pd.Timedelta(days=1)
        
        self.test_data = self.games_data[
            pd.to_datetime(self.games_data["Date"]) >= test_start_date
        ].copy()
        
        print(f"Train set: {len(self.train_data)} games")
        print(f"Test set: {len(self.test_data)} games")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Returns the final bankroll as the metric to maximize.
        """
        # Base Elo parameters (shared across all Kelly variants)
        k_factor = trial.suggest_float("k_factor", 5.0, 50.0)
        home_advantage = trial.suggest_float("home_advantage", 0.0, 200.0)
        rolling_window = trial.suggest_int("rolling_window", 3, 20)
        form_adjustment_weight = trial.suggest_float("form_adjustment_weight", 0.0, 0.3)
        streak_adjustment_weight = trial.suggest_float("streak_adjustment_weight", 0.0, 0.2)
        initial_rating = trial.suggest_float("initial_rating", 1200.0, 1800.0)
        
        # Strategy-specific parameters
        if self.strategy_type == "fractional":
            # Fractional Kelly optimization
            kelly_fraction = trial.suggest_float("kelly_fraction", 0.05, 0.5)
            min_edge = trial.suggest_float("min_edge", 0.01, 0.15)
            
            model = EloModel(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_edge=min_edge,
                kelly_fraction=kelly_fraction,
                form_adjustment_weight=form_adjustment_weight,
                streak_adjustment_weight=streak_adjustment_weight,
                initial_rating=initial_rating,
            )
            
        elif self.strategy_type == "adaptive":
            # Adaptive Kelly optimization
            base_kelly = trial.suggest_float("base_kelly", 0.1, 0.4)
            min_kelly = trial.suggest_float("min_kelly", 0.05, 0.2)
            max_kelly = trial.suggest_float("max_kelly", 0.3, 0.7)
            adaptation_rate = trial.suggest_float("adaptation_rate", 0.01, 0.1)
            min_edge = trial.suggest_float("min_edge", 0.01, 0.15)
            
            # Ensure min_kelly < base_kelly < max_kelly
            if not (min_kelly < base_kelly < max_kelly):
                raise optuna.TrialPruned()
            
            model = AdaptiveKellyStrategy(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_edge=min_edge,
                base_kelly=base_kelly,
                min_kelly=min_kelly,
                max_kelly=max_kelly,
                adaptation_rate=adaptation_rate,
                form_adjustment_weight=form_adjustment_weight,
                streak_adjustment_weight=streak_adjustment_weight,
                initial_rating=initial_rating,
            )
            
        elif self.strategy_type == "full":
            # Full Kelly optimization (mainly Elo + edge threshold)
            min_edge = trial.suggest_float("min_edge", 0.01, 0.15)
            
            model = FullKellyStrategy(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_edge=min_edge,
                form_adjustment_weight=form_adjustment_weight,
                streak_adjustment_weight=streak_adjustment_weight,
                initial_rating=initial_rating,
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Run simulation on training data
        try:
            # Suppress verbose output from Environment
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            env = Environment(
                self.train_data.copy(),
                model,
                init_bankroll=self.init_bankroll,
                min_bet=self.min_bet,
                max_bet=self.max_bet,
            )
            env.run()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            final_bankroll = env.bankroll
            
            # Prune if performance is terrible
            if final_bankroll < self.init_bankroll * 0.1:
                raise optuna.TrialPruned()
            
            return final_bankroll
            
        except Exception as e:
            # Restore stdout if there was an error
            try:
                sys.stdout = old_stdout
            except:
                pass
            print(f"Trial failed: {e}")
            raise optuna.TrialPruned()
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization and return best parameters.
        
        Returns:
            Dictionary with best parameters
        """
        print(f"\n{'='*70}")
        print(f"Optimizing {self.strategy_type.upper()} Kelly Strategy")
        print(f"{'='*70}")
        print(f"Trials: {self.n_trials}")
        print(f"Training games: {len(self.train_data)}")
        print(f"Test games: {len(self.test_data)}")
        print()
        
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        self.best_params = self.study.best_params
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best training bankroll: ${self.study.best_value:.2f}")
        print(f"\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key:30s}: {value:.4f}" if isinstance(value, float) else f"  {key:30s}: {value}")
        
        return self.best_params
    
    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Evaluate best model on test set.
        
        Returns:
            Dictionary with test metrics
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        # Create model with best parameters
        if self.strategy_type == "fractional":
            self.best_model = EloModel(**self.best_params)
        elif self.strategy_type == "adaptive":
            self.best_model = AdaptiveKellyStrategy(**self.best_params)
        elif self.strategy_type == "full":
            self.best_model = FullKellyStrategy(**self.best_params)
        
        # Run on test set
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        env = Environment(
            self.test_data.copy(),
            self.best_model,
            init_bankroll=self.init_bankroll,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
        )
        env.run()
        
        sys.stdout = old_stdout
        
        history = pd.DataFrame(env.history)
        
        # Calculate metrics
        final_bankroll = env.bankroll
        profit = final_bankroll - self.init_bankroll
        roi = (profit / self.init_bankroll) * 100
        
        # Max drawdown
        cummax = history["Bankroll"].cummax()
        drawdown = ((cummax - history["Bankroll"]) / cummax) * 100
        max_drawdown = drawdown.max() if not drawdown.empty else 0
        
        # Sharpe ratio
        returns = history["Bankroll"].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        metrics = {
            "final_bankroll": final_bankroll,
            "profit": profit,
            "roi": roi,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_bets": len(history),
        }
        
        print(f"\n{'='*70}")
        print("TEST SET EVALUATION")
        print(f"{'='*70}")
        print(f"Final Bankroll:  ${metrics['final_bankroll']:.2f}")
        print(f"Profit/Loss:     ${metrics['profit']:.2f}")
        print(f"ROI:             {metrics['roi']:.2f}%")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"Total Bets:      {metrics['total_bets']}")
        
        return metrics
    
    def save_results(self, filename: Optional[str] = None):
        """Save optimization results to file."""
        if filename is None:
            filename = f"kelly_{self.strategy_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            "strategy_type": self.strategy_type,
            "n_trials": self.n_trials,
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "optimization_history": [
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params,
                }
                for t in self.study.trials
            ],
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def optimize_all_kelly_variants(games_data: pd.DataFrame, n_trials: int = 30):
    """
    Optimize all Kelly variants and compare results.
    
    Args:
        games_data: DataFrame with game data
        n_trials: Number of trials per strategy
    
    Returns:
        Dictionary with results for each strategy
    """
    strategies = ["fractional", "adaptive", "full"]
    results = {}
    
    print(f"\n{'='*70}")
    print("OPTIMIZING ALL KELLY VARIANTS")
    print(f"{'='*70}")
    print(f"Trials per strategy: {n_trials}")
    print(f"Total games: {len(games_data)}")
    print()
    
    for strategy in strategies:
        print(f"\n{'#'*70}")
        print(f"# Strategy: {strategy.upper()}")
        print(f"{'#'*70}\n")
        
        optimizer = KellyVariantOptimizer(
            games_data=games_data,
            strategy_type=strategy,
            n_trials=n_trials,
        )
        
        best_params = optimizer.optimize()
        test_metrics = optimizer.evaluate_on_test()
        
        results[strategy] = {
            "best_params": best_params,
            "test_metrics": test_metrics,
        }
        
        # Save individual results
        optimizer.save_results()
    
    # Print comparison
    print(f"\n{'='*70}")
    print("KELLY VARIANTS COMPARISON")
    print(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame({
        strategy: {
            "Final Bankroll": f"${results[strategy]['test_metrics']['final_bankroll']:.2f}",
            "ROI": f"{results[strategy]['test_metrics']['roi']:.2f}%",
            "Max Drawdown": f"{results[strategy]['test_metrics']['max_drawdown']:.2f}%",
            "Sharpe Ratio": f"{results[strategy]['test_metrics']['sharpe_ratio']:.4f}",
        }
        for strategy in strategies
    }).T
    
    print(comparison_df)
    
    # Save comparison
    comparison_file = f"kelly_variants_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(comparison_file)
    print(f"\nComparison saved to: {comparison_file}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.insert(0, '.')
    
    # Load data
    data_path = "./data/games.csv" if os.path.exists("./data/games.csv") else "../data/games.csv"
    games = pd.read_csv(data_path, index_col=0, parse_dates=["Date", "Open"])
    print(f"Loaded {len(games)} games")
    
    # Optimize all Kelly variants
    results = optimize_all_kelly_variants(games, n_trials=30)
