"""
Enhanced evaluation script with Optuna hyperparameter optimization.

Usage:
    python src/evaluate.py                    # Run baseline evaluation
    python src/evaluate.py --optimize         # Run hyperparameter optimization
    python src/evaluate.py --optimize --trials 50  # Custom number of trials
    python src/evaluate.py --compare          # Compare different models
"""

import argparse
import pandas as pd
import logging

from model import Model
from models import EloModel, AdvancedEloModel, ConfidenceBasedEloModel
from environment import Environment
from optuna_tuner import OptunaHyperparameterTuner, compare_models, evaluate_prediction_accuracy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_baseline_evaluation(
    games: pd.DataFrame,
    init_bankroll: float = 1000.0,
    min_bet: float = 5.0,
    max_bet: float = 100.0,
):
    """Run baseline evaluation with simple random model."""
    logger.info("Running baseline evaluation...")
    
    env = Environment(
        games,
        Model(),
        init_bankroll=init_bankroll,
        min_bet=min_bet,
        max_bet=max_bet
    )
    
    env.run()
    
    print()
    print(f"Final bankroll: {env.bankroll:.2f}")
    print(f"Profit: {env.bankroll - init_bankroll:.2f}")
    print(f"ROI: {((env.bankroll - init_bankroll) / init_bankroll) * 100:.1f}%")
    
    history = env.get_history()
    return env, history


def run_optimization(
    games: pd.DataFrame,
    model_class=EloModel,
    n_trials: int = 100,
    init_bankroll: float = 1000.0,
    min_bet: float = 5.0,
    max_bet: float = 100.0,
    save_results: bool = True,
):
    """Run hyperparameter optimization."""
    logger.info(f"Starting hyperparameter optimization for {model_class.__name__}...")
    
    # Create tuner
    tuner = OptunaHyperparameterTuner(
        model_class=model_class,
        games_data=games,
        init_bankroll=init_bankroll,
        min_bet=min_bet,
        max_bet=max_bet,
        n_trials=n_trials,
        pruning=True,
        seed=42,
    )
    
    # Run optimization
    best_params = tuner.optimize(n_trials=n_trials)
    
    # Evaluate on train and test data
    train_metrics = tuner.evaluate_best_model(use_test_data=False)
    test_metrics = tuner.evaluate_best_model(use_test_data=True)
    
    # Get optimization history
    history_df = tuner.get_optimization_history()
    
    # Save results
    if save_results:
        best_params_file = f"best_params_{model_class.__name__}.json"
        tuner.save_best_model(best_params_file)
        
        history_file = f"optimization_history_{model_class.__name__}.csv"
        history_df.to_csv(history_file, index=False)
        logger.info(f"Optimization history saved to {history_file}")
        
        # Save summary
        summary_file = f"optimization_summary_{model_class.__name__}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Optimization Summary for {model_class.__name__}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of trials: {n_trials}\n")
            f.write(f"Best value (bankroll): {tuner.best_value:.2f}\n\n")
            f.write("Best Parameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nTraining Set Metrics:\n")
            for key, value in train_metrics.items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\nTest Set Metrics:\n")
            for key, value in test_metrics.items():
                f.write(f"  {key}: {value:.2f}\n")
        
        logger.info(f"Summary saved to {summary_file}")
    
    return tuner, best_params, train_metrics, test_metrics


def run_model_comparison(
    games: pd.DataFrame,
    init_bankroll: float = 1000.0,
    min_bet: float = 5.0,
    max_bet: float = 100.0,
):
    """Compare different model configurations."""
    logger.info("Running model comparison...")
    
    # Define models to compare
    model_configs = {
        "Random Model": {
            "class": Model,
            "params": {},
        },
        "EloModel (default)": {
            "class": EloModel,
            "params": {},
        },
        "EloModel (aggressive)": {
            "class": EloModel,
            "params": {
                "k_factor": 30.0,
                "min_edge": 0.03,
                "kelly_fraction": 0.4,
            },
        },
        "EloModel (conservative)": {
            "class": EloModel,
            "params": {
                "k_factor": 15.0,
                "min_edge": 0.08,
                "kelly_fraction": 0.15,
            },
        },
        "AdvancedEloModel": {
            "class": AdvancedEloModel,
            "params": {
                "score_weight": 0.15,
            },
        },
        "ConfidenceBasedElo (balanced)": {
            "class": ConfidenceBasedEloModel,
            "params": {
                "confidence_threshold": 0.15,
                "risk_factor": 0.5,
                "variance_penalty": 0.2,
                "max_bet_fraction": 0.1,
            },
        },
        "ConfidenceBasedElo (conservative)": {
            "class": ConfidenceBasedEloModel,
            "params": {
                "confidence_threshold": 0.20,
                "risk_factor": 0.3,
                "variance_penalty": 0.3,
                "max_bet_fraction": 0.05,
                "min_edge": 0.08,
                "kelly_fraction": 0.15,
            },
        },
    }
    
    results_df = compare_models(
        games,
        model_configs,
        init_bankroll=init_bankroll,
        min_bet=min_bet,
        max_bet=max_bet,
    )
    
    # Save comparison results
    results_df.to_csv("model_comparison.csv", index=False)
    logger.info("Comparison results saved to model_comparison.csv")
    
    return results_df


def test_prediction_accuracy(
    games: pd.DataFrame,
    model_class=EloModel,
    model_params: dict = None,
):
    """Test prediction accuracy of a model."""
    logger.info(f"Testing prediction accuracy for {model_class.__name__}...")
    
    if model_params is None:
        model_params = {}
    
    # Create model
    model = model_class(**model_params)
    
    # Evaluate prediction accuracy
    metrics = evaluate_prediction_accuracy(model, games)
    
    # Save results
    accuracy_file = f"prediction_accuracy_{model_class.__name__}.txt"
    with open(accuracy_file, 'w') as f:
        f.write(f"Prediction Accuracy for {model_class.__name__}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n")
        f.write(f"Home Win Accuracy: {metrics['home_accuracy']:.2%}\n")
        f.write(f"Away Win Accuracy: {metrics['away_accuracy']:.2%}\n")
        f.write(f"Draw Accuracy: {metrics['draw_accuracy']:.2%}\n")
        f.write(f"Log Loss: {metrics['log_loss']:.4f}\n")
        f.write(f"Brier Score: {metrics['brier_score']:.4f}\n")
        f.write(f"Total Predictions: {metrics['total_predictions']}\n")
    
    logger.info(f"Accuracy results saved to {accuracy_file}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate betting models with optional Optuna optimization"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["elo", "advanced_elo", "confidence"],
        default="elo",
        help="Model to optimize (default: elo)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different model configurations"
    )
    parser.add_argument(
        "--test-accuracy",
        action="store_true",
        help="Test prediction accuracy of the model"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll (default: 1000)"
    )
    parser.add_argument(
        "--min-bet",
        type=float,
        default=5.0,
        help="Minimum bet size (default: 5)"
    )
    parser.add_argument(
        "--max-bet",
        type=float,
        default=100.0,
        help="Maximum bet size (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading games data...")
    import os
    # Support running from both src/ and root directory
    data_path = "./data/games.csv" if os.path.exists("./data/games.csv") else "../data/games.csv"
    games = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=["Date", "Open"]
    )
    logger.info(f"Loaded {len(games)} games")
    
    # Execute requested operation
    if args.compare:
        run_model_comparison(
            games,
            init_bankroll=args.bankroll,
            min_bet=args.min_bet,
            max_bet=args.max_bet,
        )
    elif args.test_accuracy:
        # Select model class
        if args.model == "elo":
            model_class = EloModel
        elif args.model == "advanced_elo":
            model_class = AdvancedEloModel
        else:
            model_class = ConfidenceBasedEloModel
        
        test_prediction_accuracy(
            games,
            model_class=model_class,
        )
    elif args.optimize:
        # Select model class
        if args.model == "elo":
            model_class = EloModel
        elif args.model == "advanced_elo":
            model_class = AdvancedEloModel
        else:
            model_class = ConfidenceBasedEloModel
        
        run_optimization(
            games,
            model_class=model_class,
            n_trials=args.trials,
            init_bankroll=args.bankroll,
            min_bet=args.min_bet,
            max_bet=args.max_bet,
        )
    else:
        # Run baseline evaluation
        run_baseline_evaluation(
            games,
            init_bankroll=args.bankroll,
            min_bet=args.min_bet,
            max_bet=args.max_bet,
        )


if __name__ == "__main__":
    main()
