"""
Hyperparameter optimization using Optuna for ice hockey betting model.
Optimizes model accuracy metrics.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Import your model
import sys
sys.path.insert(0, '.')
from src.model import Model
from src.environment import Environment


class ModelOptimizer:
    """Optimize model hyperparameters using Optuna"""
    
    def __init__(self, games_path='./data/games.csv', validation_period=365):
        """
        Initialize optimizer
        
        Args:
            games_path: Path to games CSV
            validation_period: Days to use for validation (rest is training)
        """
        print("Loading data...")
        self.games = pd.read_csv(games_path, index_col=0, parse_dates=["Date", "Open"])
        print(f"Loaded {len(self.games)} games")
        
        # Define train/val split
        max_date = self.games['Date'].max()
        self.val_start = max_date - pd.Timedelta(days=validation_period)
        self.train_end = self.val_start
        self.train_start = self.games['Date'].min()
        
        print(f"\nTrain: {self.train_start.date()} to {self.train_end.date()}")
        print(f"Val:   {self.val_start.date()} to {max_date.date()}")
        
    def objective(self, trial):
        """
        Objective function for Optuna
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Metric to optimize (we'll use negative log-loss, so higher is better)
        """
        # Suggest hyperparameters
        # These are parameters you can tune in your model
        params = {
            'ensamble_required_n': trial.suggest_int('ensamble_required_n', 300, 1000, step=100),
            # Add more tunable parameters from your Model class
            # Examples:
            # 'pythagorean_power': trial.suggest_float('pythagorean_power', 0.8, 1.5),
            # 'elo_k_factor': trial.suggest_int('elo_k_factor', 10, 25),
            # etc.
        }
        
        try:
            # Create model with these parameters
            model = self._create_model_with_params(params)
            
            # Evaluate model
            metrics = self._evaluate_model(model)
            
            # Log metrics
            trial.set_user_attr('accuracy', metrics['accuracy'])
            trial.set_user_attr('brier_score', metrics['brier_score'])
            trial.set_user_attr('n_predictions', metrics['n_predictions'])
            
            # Return negative log-loss (Optuna maximizes by default)
            return -metrics['log_loss']
        
        except Exception as e:
            print(f"Error in trial: {e}")
            return -999.0  # Large penalty for failed trials
    
    def _create_model_with_params(self, params):
        """Create a model instance with given parameters"""
        # For now, Model() doesn't take parameters
        # You'll need to modify this to pass parameters to your model
        # For example:
        # return Model(ensamble_required_n=params['ensamble_required_n'])
        
        model = Model()
        
        # Set parameters after initialization
        if 'ensamble_required_n' in params:
            model.ensamble_required_n = params['ensamble_required_n']
        
        return model
    
    def _evaluate_model(self, model):
        """Evaluate model on validation set"""
        # Create environment
        env = Environment(
            self.games.copy(),
            model,
            start_date=self.train_start,
            end_date=self.games['Date'].max(),
            init_bankroll=1000,
            min_bet=5,
            max_bet=100
        )
        
        # Run simulation and collect predictions
        predictions = []
        
        for date in pd.date_range(self.train_start, self.games['Date'].max()):
            inc = env._next_date(date)
            opps = env._get_options(date)
            
            if opps.empty and inc.empty:
                continue
            
            summary = env._generate_summary(date)
            bets = model.place_bets(summary, opps, inc)
            
            # Track validation predictions
            if date >= self.val_start:
                for idx in opps.index:
                    if idx in model.prediction_map:
                        game = self.games.loc[idx]
                        pred = model.prediction_map[idx]
                        
                        predictions.append({
                            'pred_home': pred,
                            'actual_home': game['H']
                        })
            
            validated_bets = env._validate_bets(bets, opps)
            env._place_bets(date, validated_bets)
        
        # Calculate metrics
        return self._calculate_metrics(predictions)
    
    def _calculate_metrics(self, predictions):
        """Calculate prediction metrics"""
        if not predictions:
            return {
                'n_predictions': 0,
                'accuracy': 0.0,
                'log_loss': 999.0,
                'brier_score': 1.0
            }
        
        n = len(predictions)
        correct = 0
        log_loss = 0.0
        brier_score = 0.0
        
        for pred in predictions:
            p = pred['pred_home']
            actual = pred['actual_home']
            
            # Accuracy
            if (p > 0.5 and actual == 1) or (p < 0.5 and actual == 0):
                correct += 1
            
            # Log loss
            p_clamped = max(0.001, min(0.999, p))
            log_loss += -(actual * np.log(p_clamped) + (1 - actual) * np.log(1 - p_clamped))
            
            # Brier score
            brier_score += (p - actual) ** 2
        
        return {
            'n_predictions': n,
            'accuracy': correct / n,
            'log_loss': log_loss / n,
            'brier_score': brier_score / n
        }
    
    def optimize(self, n_trials=50, study_name='hockey_model_optimization'):
        """
        Run optimization
        
        Args:
            n_trials: Number of trials to run
            study_name: Name for the study
        
        Returns:
            Best parameters
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        print(f"Study: {study_name}")
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize negative log-loss
            sampler=TPESampler(seed=42),
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Print results
        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        
        print(f"\nBest trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")
        print(f"  Accuracy: {study.best_trial.user_attrs.get('accuracy', 'N/A'):.4f}")
        print(f"  Brier: {study.best_trial.user_attrs.get('brier_score', 'N/A'):.4f}")
        print(f"  N: {study.best_trial.user_attrs.get('n_predictions', 'N/A')}")
        
        # Print top 5 trials
        print(f"\nTop 5 trials:")
        for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]):
            print(f"\n{i+1}. Trial {trial.number}:")
            print(f"   Value: {trial.value:.4f}")
            print(f"   Params: {trial.params}")
            print(f"   Accuracy: {trial.user_attrs.get('accuracy', 'N/A'):.4f}")
        
        # Save results
        results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'best_attrs': study.best_trial.user_attrs,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'attrs': t.user_attrs
                }
                for t in study.trials
            ]
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to optimization_results.json")
        
        return study.best_trial.params


def main():
    """Run optimization"""
    optimizer = ModelOptimizer(validation_period=365)
    
    # Run optimization with 50 trials (adjust as needed)
    best_params = optimizer.optimize(n_trials=50)
    
    print(f"\n{'='*60}")
    print("FINAL BEST PARAMETERS")
    print(f"{'='*60}")
    print(json.dumps(best_params, indent=2))


if __name__ == '__main__':
    main()
