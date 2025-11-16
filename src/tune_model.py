"""
Model tuning framework for ice hockey betting model.
Focuses on prediction accuracy first, betting strategy optimization later.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from model import Model
from environment import Environment


class ModelTuner:
    """Framework for tuning model hyperparameters"""
    
    def __init__(self, games_path='./data/games.csv'):
        """Initialize tuner with data"""
        print("Loading data...")
        self.games = pd.read_csv(games_path, index_col=0, parse_dates=["Date", "Open"])
        print(f"Loaded {len(self.games)} games")
        
    def create_time_based_splits(self, n_folds=5, test_size=0.2):
        """
        Create time-based train/validation splits (no shuffling for time series)
        
        Args:
            n_folds: Number of validation folds
            test_size: Fraction of data to use for final test
        
        Returns:
            List of (train_dates, val_dates, test_dates) tuples
        """
        # Sort by date
        sorted_games = self.games.sort_values('Date')
        
        # Get date range
        min_date = sorted_games['Date'].min()
        max_date = sorted_games['Date'].max()
        
        # Split into train+val (80%) and test (20%)
        date_range = (max_date - min_date).days
        test_cutoff = max_date - pd.Timedelta(days=int(date_range * test_size))
        
        train_val_games = sorted_games[sorted_games['Date'] < test_cutoff]
        
        # Create rolling window splits for cross-validation
        splits = []
        train_val_range = (train_val_games['Date'].max() - train_val_games['Date'].min()).days
        fold_size = train_val_range // (n_folds + 1)
        
        for i in range(n_folds):
            # Use expanding window: train on everything before validation period
            val_start = train_val_games['Date'].min() + pd.Timedelta(days=(i + 1) * fold_size)
            val_end = val_start + pd.Timedelta(days=fold_size)
            
            train_end = val_start
            
            splits.append({
                'train_start': train_val_games['Date'].min(),
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'test_start': test_cutoff,
                'test_end': max_date
            })
        
        return splits
    
    def evaluate_model(self, model_class, params, split, return_predictions=False):
        """
        Evaluate model on a single train/val split
        
        Args:
            model_class: Model class to instantiate
            params: Dictionary of hyperparameters
            split: Dictionary with train/val/test date ranges
            return_predictions: Whether to return individual predictions
        
        Returns:
            Dictionary with metrics
        """
        # Create model with custom parameters
        model = model_class(**params)
        
        # Create environment for this split
        env = Environment(
            self.games.copy(),
            model,
            start_date=split['train_start'],
            end_date=split['val_end'],
            init_bankroll=1000,
            min_bet=5,
            max_bet=100
        )
        
        # Track predictions during validation period
        predictions = []
        
        # Run simulation
        for date in pd.date_range(split['train_start'], split['val_end']):
            inc = env._next_date(date)
            opps = env._get_options(date)
            
            if opps.empty and inc.empty:
                continue
            
            summary = env._generate_summary(date)
            bets = model.place_bets(summary, opps, inc)
            
            # Track predictions made during validation period
            if date >= split['val_start'] and date < split['val_end']:
                for idx in opps.index:
                    if idx in model.prediction_map:
                        game = self.games.loc[idx]
                        pred = model.prediction_map[idx]
                        
                        predictions.append({
                            'date': game['Date'],
                            'home_id': game['HID'],
                            'away_id': game['AID'],
                            'pred_home': pred,
                            'actual_home': game['H'],
                            'actual_away': game['A'],
                            'actual_draw': game['D'],
                            'odds_home': game['OddsH'],
                            'odds_away': game['OddsA'],
                            'odds_draw': game['OddsD']
                        })
            
            validated_bets = env._validate_bets(bets, opps)
            env._place_bets(date, validated_bets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions)
        
        if return_predictions:
            return metrics, predictions
        return metrics
    
    def _calculate_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate prediction quality metrics"""
        if not predictions:
            return {
                'n_predictions': 0,
                'accuracy': 0.0,
                'log_loss': 999.0,
                'brier_score': 1.0,
                'calibration_error': 1.0
            }
        
        n = len(predictions)
        correct = 0
        log_loss = 0.0
        brier_score = 0.0
        
        # Calibration bins
        n_bins = 10
        bin_totals = defaultdict(int)
        bin_correct = defaultdict(int)
        
        for pred in predictions:
            p = pred['pred_home']
            actual = pred['actual_home']
            
            # Accuracy (binary)
            if (p > 0.5 and actual == 1) or (p < 0.5 and actual == 0):
                correct += 1
            
            # Log loss (cross-entropy)
            p_clamped = max(0.001, min(0.999, p))
            log_loss += -(actual * np.log(p_clamped) + (1 - actual) * np.log(1 - p_clamped))
            
            # Brier score (MSE of probabilities)
            brier_score += (p - actual) ** 2
            
            # Calibration
            bin_idx = min(int(p * n_bins), n_bins - 1)
            bin_totals[bin_idx] += 1
            bin_correct[bin_idx] += actual
        
        # Expected calibration error
        ece = 0.0
        for bin_idx in range(n_bins):
            if bin_totals[bin_idx] > 0:
                bin_accuracy = bin_correct[bin_idx] / bin_totals[bin_idx]
                bin_confidence = (bin_idx + 0.5) / n_bins
                ece += abs(bin_accuracy - bin_confidence) * bin_totals[bin_idx] / n
        
        return {
            'n_predictions': n,
            'accuracy': correct / n,
            'log_loss': log_loss / n,
            'brier_score': brier_score / n,
            'calibration_error': ece
        }
    
    def cross_validate(self, model_class, params, n_folds=3):
        """
        Cross-validate model with given parameters
        
        Args:
            model_class: Model class to instantiate
            params: Dictionary of hyperparameters
            n_folds: Number of CV folds
        
        Returns:
            Dictionary with mean and std of metrics
        """
        print(f"\nCross-validating with {n_folds} folds...")
        print(f"Parameters: {params}")
        
        splits = self.create_time_based_splits(n_folds=n_folds)
        
        all_metrics = []
        for i, split in enumerate(splits):
            print(f"\nFold {i+1}/{n_folds}")
            print(f"  Train: {split['train_start'].date()} to {split['train_end'].date()}")
            print(f"  Val:   {split['val_start'].date()} to {split['val_end'].date()}")
            
            metrics = self.evaluate_model(model_class, params, split)
            all_metrics.append(metrics)
            
            print(f"  Results: acc={metrics['accuracy']:.4f}, "
                  f"log_loss={metrics['log_loss']:.4f}, "
                  f"brier={metrics['brier_score']:.4f}, "
                  f"n={metrics['n_predictions']}")
        
        # Aggregate results
        mean_metrics = {}
        std_metrics = {}
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            mean_metrics[f'{key}_mean'] = np.mean(values)
            std_metrics[f'{key}_std'] = np.std(values)
        
        return {**mean_metrics, **std_metrics}
    
    def grid_search(self, model_class, param_grid, n_folds=3, metric='accuracy'):
        """
        Grid search over hyperparameters
        
        Args:
            model_class: Model class to instantiate
            param_grid: Dictionary mapping param names to lists of values
            n_folds: Number of CV folds
            metric: Metric to optimize ('accuracy', 'log_loss', 'brier_score')
        
        Returns:
            Best parameters and results
        """
        print(f"\n{'='*60}")
        print("GRID SEARCH")
        print(f"{'='*60}")
        print(f"Optimizing: {metric}")
        print(f"Parameter grid: {param_grid}")
        
        # Generate all parameter combinations
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\nTesting {len(combinations)} parameter combinations...")
        
        results = []
        for i, params in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            try:
                cv_results = self.cross_validate(model_class, params, n_folds)
                
                result = {
                    'params': params,
                    'cv_results': cv_results,
                    'score': cv_results[f'{metric}_mean']
                }
                results.append(result)
                
                print(f"  Score ({metric}): {result['score']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        # Sort by metric (lower is better for log_loss and brier_score)
        reverse = metric == 'accuracy'
        results.sort(key=lambda x: x['score'], reverse=reverse)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        
        for i, result in enumerate(results[:5]):
            print(f"\nRank {i+1}:")
            print(f"  Params: {result['params']}")
            print(f"  {metric}: {result['score']:.4f}")
            print(f"  All metrics: {result['cv_results']}")
        
        # Save results
        with open('tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to tuning_results.json")
        
        return results[0]['params'], results
    
    def evaluate_on_test_set(self, model_class, params):
        """
        Final evaluation on held-out test set
        
        Args:
            model_class: Model class to instantiate
            params: Dictionary of hyperparameters
        
        Returns:
            Test set metrics
        """
        print(f"\n{'='*60}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*60}")
        
        splits = self.create_time_based_splits(n_folds=1)
        split = splits[0]
        
        print(f"Train: {split['train_start'].date()} to {split['train_end'].date()}")
        print(f"Test:  {split['test_start'].date()} to {split['test_end'].date()}")
        
        # Evaluate on full train -> test
        test_split = {
            'train_start': split['train_start'],
            'train_end': split['test_start'],
            'val_start': split['test_start'],
            'val_end': split['test_end'],
            'test_start': split['test_start'],
            'test_end': split['test_end']
        }
        
        metrics, predictions = self.evaluate_model(
            model_class, params, test_split, return_predictions=True
        )
        
        print(f"\nTest Set Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save predictions for analysis
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('test_predictions.csv', index=False)
        print(f"\nPredictions saved to test_predictions.csv")
        
        return metrics, predictions


def main():
    """Example usage"""
    tuner = ModelTuner()
    
    # Example: Grid search over model parameters
    # You can add parameters from your Model class here
    param_grid = {
        # Example parameters - adjust based on what you want to tune
        # 'ensamble_required_n': [300, 500, 700],
        # Add more parameters as needed
    }
    
    # For now, just use default parameters
    default_params = {}
    
    print("Running cross-validation with default parameters...")
    cv_results = tuner.cross_validate(Model, default_params, n_folds=3)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for key, value in cv_results.items():
        print(f"{key}: {value:.4f}")
    
    # Optionally run grid search
    # best_params, all_results = tuner.grid_search(Model, param_grid, n_folds=3)
    
    # Final evaluation
    # test_metrics, predictions = tuner.evaluate_on_test_set(Model, default_params)


if __name__ == '__main__':
    main()
