"""
Hyperparameter optimization using Optuna for ice hockey betting model.
Optimizes model accuracy metrics with parallel execution support.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import argparse
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
        # Suggest hyperparameters - focusing on most impactful ones
        params = {
            # Ensemble parameters
            'ensamble_required_n': trial.suggest_int('ensamble_required_n', 300, 1000, step=100),
            'ensamble_retrain_freq': trial.suggest_int('ensamble_retrain_freq', 200, 800, step=100),
            
            # Pythagorean model
            'pythagorean_power': trial.suggest_float('pythagorean_power', 0.8, 1.5),
            'pythagorean_regularization': trial.suggest_int('pythagorean_regularization', 10, 40, step=5),
            
            # Four Factor model
            'four_factor_required_n': trial.suggest_int('four_factor_required_n', 300, 1000, step=100),
            'four_factor_decay': trial.suggest_float('four_factor_decay', 0.990, 0.998),
            
            # Gradient Descent
            'gradient_descent_epochs': trial.suggest_int('gradient_descent_epochs', 8, 20),
            'gradient_descent_min_games': trial.suggest_int('gradient_descent_min_games', 300, 800, step=100),
            
            # Exhaustion model
            'exhaustion_b2b_penalty': trial.suggest_float('exhaustion_b2b_penalty', 1.0, 2.5),
            'exhaustion_road_penalty': trial.suggest_float('exhaustion_road_penalty', 0.3, 0.8),
            
            # ELO system
            'elo_k_base': trial.suggest_int('elo_k_base', 10, 25),
            'elo_home_advantage': trial.suggest_int('elo_home_advantage', 25, 50, step=5),
            'elo_margin_cap': trial.suggest_int('elo_margin_cap', 2, 5),
            
            # Poisson model
            'poisson_regularization': trial.suggest_float('poisson_regularization', 2.0, 5.0),
            
            # Rolling window features (NEW)
            'rolling_window_short': trial.suggest_int('rolling_window_short', 3, 7),
            'rolling_window_med': trial.suggest_int('rolling_window_med', 8, 15),
            'rolling_window_long': trial.suggest_int('rolling_window_long', 16, 25),
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
        return Model(**params)
    
    def _evaluate_model(self, model):
        """Evaluate model on validation set - focuses on prediction accuracy only"""
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
            
            # Process incremental data to update model state
            for idx in inc.index:
                current = inc.loc[idx]
                model._game_increment(idx, current)
            
            # Make predictions for ALL games in validation period (not just bets)
            if date >= self.val_start:
                for idx in opps.index:
                    current = opps.loc[idx]
                    season = current['Season']
                    game_date = current['Date']
                    home_id = current['HID']
                    away_id = current['AID']
                    
                    # Only predict if ensemble is trained
                    if len(model.past_pred) >= model.ensamble_required_n:
                        input_arr = model._get_input_features(home_id, away_id, season, game_date)
                        
                        if input_arr is not None:
                            # Retrain ensemble if needed
                            if model.ensamble_retrain <= 0:
                                model.ensamble_retrain = model.ensamble_retrain_freq
                                np_array = np.array(model.past_pred)
                                sample_weights = np.exp(-0.0003 * np.arange(len(model.past_pred)))
                                from sklearn.linear_model import LogisticRegression
                                model.ensamble = LogisticRegression(
                                    max_iter=10000,
                                    multi_class='multinomial',
                                    solver='lbfgs'
                                )
                                model.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
                            
                            # Get 3-class prediction from ensemble: [P(Away), P(Draw), P(Home)]
                            probs = model.ensamble.predict_proba(np.array([input_arr]))[0]
                            
                            # Store prediction for evaluation
                            game = self.games.loc[idx]
                            predictions.append({
                                'prob_away': probs[0],
                                'prob_draw': probs[1],
                                'prob_home': probs[2],
                                'actual_away': game['A'],
                                'actual_draw': game['D'],
                                'actual_home': game['H']
                            })
        
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
            p_away = pred['prob_away']
            p_draw = pred['prob_draw']
            p_home = pred['prob_home']
            actual_away = pred['actual_away']
            actual_draw = pred['actual_draw']
            actual_home = pred['actual_home']
            
            # Accuracy - predicted class matches actual
            pred_class = np.argmax([p_away, p_draw, p_home])  # 0=Away, 1=Draw, 2=Home
            if pred_class == 0 and actual_away:
                correct += 1
            elif pred_class == 1 and actual_draw:
                correct += 1
            elif pred_class == 2 and actual_home:
                correct += 1
            
            # Log loss (multinomial cross-entropy)
            p_away_clamped = max(0.001, min(0.999, p_away))
            p_draw_clamped = max(0.001, min(0.999, p_draw))
            p_home_clamped = max(0.001, min(0.999, p_home))
            
            log_loss += -(
                actual_away * np.log(p_away_clamped) +
                actual_draw * np.log(p_draw_clamped) +
                actual_home * np.log(p_home_clamped)
            )
            
            # Brier score (mean squared error across all classes)
            brier_score += (
                (p_away - actual_away) ** 2 +
                (p_draw - actual_draw) ** 2 +
                (p_home - actual_home) ** 2
            ) / 3  # Average over 3 classes
        
        return {
            'n_predictions': n,
            'accuracy': correct / n,
            'log_loss': log_loss / n,
            'brier_score': brier_score / n
        }
    
    def optimize(self, n_trials=50, study_name='hockey_model_optimization', n_jobs=1, storage=None):
        """
        Run optimization with optional parallel execution
        
        Args:
            n_trials: Number of trials to run
            study_name: Name for the study
            n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
            storage: Optuna storage URL (e.g., 'sqlite:///optuna_study.db')
                    Required for n_jobs > 1
        
        Returns:
            Best parameters
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        print(f"Study: {study_name}")
        print(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all CPUs'}")
        
        # Setup storage for parallel execution
        if n_jobs != 1 and storage is None:
            storage = f'sqlite:///{study_name}.db'
            print(f"Storage: {storage}")
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize negative log-loss
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=True
        )
        
        # Optimize with parallel execution
        if n_jobs == 1:
            # Sequential execution
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        else:
            # Parallel execution
            print(f"\nRunning {n_trials} trials across {n_jobs if n_jobs > 0 else 'all'} workers...")
            study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
        
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
    """Run optimization with command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimize hockey betting model hyperparameters with parallel execution support'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs. Use -1 for all CPUs, 1 for sequential (default: 1)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='hockey_model_optimization',
        help='Name for the Optuna study (default: hockey_model_optimization)'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Optuna storage URL (e.g., sqlite:///optuna.db). Auto-generated if n-jobs > 1'
    )
    parser.add_argument(
        '--validation-period',
        type=int,
        default=365,
        help='Validation period in days (default: 365)'
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(validation_period=args.validation_period)
    
    # Run optimization
    best_params = optimizer.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        storage=args.storage
    )
    
    print(f"\n{'='*60}")
    print("FINAL BEST PARAMETERS")
    print(f"{'='*60}")
    print(json.dumps(best_params, indent=2))


if __name__ == '__main__':
    main()
