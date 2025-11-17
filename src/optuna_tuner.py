"""
Optuna-based hyperparameter optimization for betting models.

This module provides:
- OptunaHyperparameterTuner: Framework for optimizing model hyperparameters
- Cross-validation support with train/test splits
- Pruning strategies for efficient optimization
- Results logging and analysis
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Type
import logging

from environment import Environment
from models import BaseModel, EloModel, AdvancedEloModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaHyperparameterTuner:
    """
    Optuna-based hyperparameter optimizer for betting models.
    
    Uses Bayesian optimization (TPE sampler) to find optimal hyperparameters
    that maximize final bankroll/profit.
    """
    
    def __init__(
        self,
        model_class: Type[BaseModel],
        games_data: pd.DataFrame,
        train_end_date: Optional[str] = None,
        test_start_date: Optional[str] = None,
        init_bankroll: float = 1000.0,
        min_bet: float = 5.0,
        max_bet: float = 100.0,
        n_trials: int = 100,
        pruning: bool = True,
        seed: Optional[int] = 42,
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_class: Model class to optimize
            games_data: Full games dataset
            train_end_date: End date for training period (cross-validation)
            test_start_date: Start date for test period (validation)
            init_bankroll: Initial bankroll for simulation
            min_bet: Minimum bet size
            max_bet: Maximum bet size
            n_trials: Number of optimization trials
            pruning: Whether to use pruning for efficiency
            seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.games_data = games_data.copy()
        self.init_bankroll = init_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.n_trials = n_trials
        self.seed = seed
        
        # Set up train/test split
        if train_end_date is None:
            # Use first 80% for training
            all_dates = games_data["Date"].sort_values()
            split_idx = int(len(all_dates) * 0.8)
            train_end_date = all_dates.iloc[split_idx]
        
        if test_start_date is None:
            test_start_date = train_end_date
        
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        
        # Split data
        self.train_data = games_data[games_data["Date"] <= self.train_end_date]
        self.test_data = games_data[games_data["Date"] >= self.test_start_date]
        
        logger.info(f"Train period: {self.train_data['Date'].min()} to {self.train_data['Date'].max()}")
        logger.info(f"Test period: {self.test_data['Date'].min()} to {self.test_data['Date'].max()}")
        logger.info(f"Train games: {len(self.train_data)}, Test games: {len(self.test_data)}")
        
        # Set up Optuna study
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner() if pruning else optuna.pruners.NopPruner()
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_class.__name__}_optimization",
        )
        
        self.best_params = None
        self.best_value = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Suggests hyperparameters, runs simulation, and returns final bankroll.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Final bankroll (profit metric to maximize)
        """
        # Suggest hyperparameters based on model type
        params = self._suggest_hyperparameters(trial)
        
        try:
            # Create model with suggested parameters
            model = self.model_class(**params)
            
            # Run simulation on training data
            env = Environment(
                self.train_data.copy(),
                model,
                init_bankroll=self.init_bankroll,
                min_bet=self.min_bet,
                max_bet=self.max_bet,
            )
            
            env.run()
            final_bankroll = env.bankroll
            
            # Log trial result
            profit = final_bankroll - self.init_bankroll
            roi = (profit / self.init_bankroll) * 100
            
            logger.info(
                f"Trial {trial.number}: Bankroll={final_bankroll:.2f}, "
                f"Profit={profit:.2f}, ROI={roi:.1f}%"
            )
            
            # Optionally report intermediate values for pruning
            # (could use bankroll at different time points)
            trial.report(final_bankroll, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return final_bankroll
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Return initial bankroll (no profit/loss) for failed trials
            return self.init_bankroll
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the model.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        if self.model_class == EloModel or issubclass(self.model_class, EloModel):
            params = {
                "k_factor": trial.suggest_float("k_factor", 5.0, 50.0),
                "home_advantage": trial.suggest_float("home_advantage", 0.0, 200.0),
                "rolling_window": trial.suggest_int("rolling_window", 3, 20),
                "min_edge": trial.suggest_float("min_edge", 0.01, 0.15),
                "kelly_fraction": trial.suggest_float("kelly_fraction", 0.05, 0.5),
                "form_adjustment_weight": trial.suggest_float(
                    "form_adjustment_weight", 0.0, 0.3
                ),
                "streak_adjustment_weight": trial.suggest_float(
                    "streak_adjustment_weight", 0.0, 0.2
                ),
                "initial_rating": trial.suggest_float("initial_rating", 1200.0, 1800.0),
            }
            
            # Add AdvancedEloModel specific parameters
            if self.model_class == AdvancedEloModel:
                params["score_weight"] = trial.suggest_float("score_weight", 0.0, 0.3)
            
            return params
        else:
            # Default parameter space - can be extended for other model types
            return {}
    
    def optimize(self, n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials (uses instance default if None)
            
        Returns:
            Dictionary of best hyperparameters
        """
        if n_trials is None:
            n_trials = self.n_trials
        
        logger.info(f"Starting optimization with {n_trials} trials...")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best value (bankroll): {self.best_value:.2f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def evaluate_best_model(self, use_test_data: bool = True) -> Dict[str, float]:
        """
        Evaluate the best model on test data.
        
        Args:
            use_test_data: Whether to use test data (True) or training data (False)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before evaluating")
        
        data = self.test_data if use_test_data else self.train_data
        dataset_name = "test" if use_test_data else "train"
        
        logger.info(f"\nEvaluating best model on {dataset_name} data...")
        
        # Create model with best parameters
        model = self.model_class(**self.best_params)
        
        # Run simulation
        env = Environment(
            data.copy(),
            model,
            init_bankroll=self.init_bankroll,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
        )
        
        env.run()
        
        # Calculate metrics
        final_bankroll = env.bankroll
        profit = final_bankroll - self.init_bankroll
        roi = (profit / self.init_bankroll) * 100
        
        history = env.get_history()
        
        # Calculate additional metrics
        max_bankroll = history["Bankroll"].max()
        min_bankroll = history["Bankroll"].min()
        max_drawdown = ((max_bankroll - min_bankroll) / max_bankroll) * 100
        
        metrics = {
            "final_bankroll": final_bankroll,
            "profit": profit,
            "roi": roi,
            "max_bankroll": max_bankroll,
            "min_bankroll": min_bankroll,
            "max_drawdown": max_drawdown,
        }
        
        logger.info(f"\n{dataset_name.upper()} SET RESULTS:")
        logger.info(f"Final Bankroll: {final_bankroll:.2f}")
        logger.info(f"Profit: {profit:.2f}")
        logger.info(f"ROI: {roi:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        
        return metrics
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with trial results
        """
        trials_df = self.study.trials_dataframe()
        return trials_df
    
    def plot_optimization_history(self) -> None:
        """Plot optimization history (requires plotly)."""
        try:
            import optuna.visualization as vis
            
            # Plot optimization history
            fig1 = vis.plot_optimization_history(self.study)
            fig1.show()
            
            # Plot parameter importances
            fig2 = vis.plot_param_importances(self.study)
            fig2.show()
            
            # Plot parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(self.study)
            fig3.show()
            
        except ImportError:
            logger.warning(
                "plotly not installed. Install with: pip install plotly"
            )
    
    def save_best_model(self, filepath: str) -> None:
        """
        Save best model parameters to file.
        
        Args:
            filepath: Path to save parameters (JSON format)
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before saving")
        
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {filepath}")
    
    def load_and_create_model(self, filepath: str) -> BaseModel:
        """
        Load parameters and create model.
        
        Args:
            filepath: Path to parameters file (JSON format)
            
        Returns:
            Model instance with loaded parameters
        """
        import json
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        model = self.model_class(**params)
        logger.info(f"Model created with parameters from {filepath}")
        
        return model


def compare_models(
    games_data: pd.DataFrame,
    model_configs: Dict[str, Dict[str, Any]],
    init_bankroll: float = 1000.0,
    min_bet: float = 5.0,
    max_bet: float = 100.0,
) -> pd.DataFrame:
    """
    Compare multiple model configurations.
    
    Args:
        games_data: Full games dataset
        model_configs: Dictionary mapping model names to config dicts with
                      'class' and 'params' keys
        init_bankroll: Initial bankroll
        min_bet: Minimum bet size
        max_bet: Maximum bet size
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, config in model_configs.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        model_class = config["class"]
        params = config.get("params", {})
        
        # Create and evaluate model
        model = model_class(**params)
        env = Environment(
            games_data.copy(),
            model,
            init_bankroll=init_bankroll,
            min_bet=min_bet,
            max_bet=max_bet,
        )
        
        env.run()
        
        # Calculate metrics
        final_bankroll = env.bankroll
        profit = final_bankroll - init_bankroll
        roi = (profit / init_bankroll) * 100
        
        history = env.get_history()
        max_bankroll = history["Bankroll"].max()
        min_bankroll = history["Bankroll"].min()
        max_drawdown = ((max_bankroll - min_bankroll) / max_bankroll) * 100 if max_bankroll > 0 else 0
        
        results.append({
            "Model": model_name,
            "Final_Bankroll": final_bankroll,
            "Profit": profit,
            "ROI": roi,
            "Max_Drawdown": max_drawdown,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Final_Bankroll", ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(results_df.to_string(index=False))
    
    return results_df
