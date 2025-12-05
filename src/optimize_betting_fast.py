"""
Fast Betting Strategy Optimization - FIXED

Pre-trains the model once, then only optimizes betting parameters.
This is 10-60x faster than retraining on every trial.

FIXED: Properly collects predictions by simulating the full environment flow.
"""

import argparse
import pandas as pd
import numpy as np
import optuna
import pickle
import os
from tqdm import tqdm

from model_simple_rolling import Model
from environment import Environment


def train_model_and_cache_predictions(games):
    """
    Run full environment simulation once and cache all predictions.
    
    Returns:
        dict with:
            - 'predictions': {game_id: prob_home_win}
            - 'games': DataFrame of games
    """
    print("Training model and caching predictions...")
    print("This simulates the full betting environment once.\n")
    
    # Create model with default parameters
    model = Model(
        rolling_window_short=5,
        rolling_window_med=10,
        rolling_window_long=20,
        required_games=100,
        retrain_freq=50
    )
    
    # Run environment to get all predictions
    env = Environment(
        games=games.copy(),
        model=model,
        init_bankroll=1000.0,
        min_bet=5,
        max_bet=100
    )
    
    # Collect predictions during environment run
    predictions = {}
    
    # Monkey-patch the model's place_bets to capture predictions
    original_place_bets = model.place_bets
    
    def place_bets_with_capture(summary, opps, inc):
        # Capture predictions for all opportunities
        for i in opps.index:
            current = opps.loc[i]
            home_id = current['HID']
            away_id = current['AID']
            
            features = model.rolling.get_input_data(home_id, away_id)
            
            if features is not None and model.model is not None:
                prob_home_win = model.model.predict_proba(
                    model.scaler.transform([features])
                )[0][1]
                predictions[i] = prob_home_win
        
        # Call original place_bets
        return original_place_bets(summary, opps, inc)
    
    model.place_bets = place_bets_with_capture
    
    # Run environment
    print("Running environment simulation...")
    env.run()
    
    print(f"✓ Cached {len(predictions)} predictions")
    print(f"✓ Final bankroll: ${env.bankroll:.2f}\n")
    
    return {
        'predictions': predictions,
        'games': games
    }


def objective_with_cached_predictions(trial, cached_data):
    """
    Objective function using pre-computed predictions.
    Only optimizes betting parameters.
    """
    predictions = cached_data['predictions']
    games = cached_data['games']
    
    # Suggest betting strategy parameters
    min_edge = trial.suggest_float('min_edge', 0.0, 0.20)
    max_kelly_fraction = trial.suggest_float('max_kelly_fraction', 0.05, 0.50)
    kelly_multiplier = trial.suggest_float('kelly_multiplier', 0.2, 2.0)
    normalize_odds = trial.suggest_categorical('normalize_odds', [True, False])
    min_probability = trial.suggest_float('min_probability', 0.45, 0.95)
    
    # Simulate betting with cached predictions
    bankroll = 1000.0
    min_bet = 5
    max_bet = 100
    bets_placed = 0
    
    for idx in predictions.keys():
        current = games.loc[idx]
        prob_home_win = predictions[idx]
        
        # Calculate edge
        odds_home = current['OddsH']
        
        if normalize_odds:
            odds_away = current['OddsA']
            odds_draw = current.get('OddsD', 0)
            
            raw_prob_home = 1.0 / odds_home
            raw_prob_away = 1.0 / odds_away
            raw_prob_draw = 1.0 / odds_draw if odds_draw > 0 else 0
            
            total = raw_prob_home + raw_prob_away + raw_prob_draw
            bookmaker_prob = raw_prob_home / total
        else:
            bookmaker_prob = 1.0 / odds_home
        
        edge = prob_home_win - bookmaker_prob
        
        # Check betting criteria
        if edge > min_edge and prob_home_win > min_probability:
            # Calculate bet size
            kelly_fraction = (edge / (odds_home - 1)) * kelly_multiplier
            safe_fraction = max(0, min(kelly_fraction, max_kelly_fraction))
            
            bet_amount = safe_fraction * bankroll
            bet_amount = np.round(max(min_bet, min(bet_amount, max_bet)))
            
            # Skip if Kelly suggests less than min_bet
            if bet_amount < min_bet:
                continue
            
            bets_placed += 1
            
            # Update bankroll based on outcome
            if current['H']:  # Home win
                bankroll += bet_amount * (odds_home - 1)
            else:  # Home loss
                bankroll -= bet_amount
    
    profit = bankroll - 1000.0
    return profit


def main():
    parser = argparse.ArgumentParser(description='Fast betting strategy optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--cache-file', type=str, default='model_predictions_cache.pkl',
                       help='Cache file for predictions')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain even if cache exists')
    parser.add_argument('--study-name', type=str, default='fast_betting_opt',
                       help='Name of the optimization study')
    
    args = parser.parse_args()
    
    # Load or create cache
    if not args.retrain and os.path.exists(args.cache_file):
        print(f"Loading cached predictions from {args.cache_file}...")
        with open(args.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"✓ Loaded {len(cached_data['predictions'])} cached predictions\n")
    else:
        games = pd.read_csv("../data/games.csv", index_col=0, parse_dates=["Date", "Open"])
        cached_data = train_model_and_cache_predictions(games)
        
        # Save cache
        print(f"Saving cache to {args.cache_file}...")
        with open(args.cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"✓ Cached predictions saved\n")
    
    # Setup storage for parallel execution
    storage = None
    if args.n_jobs != 1:
        storage = f'sqlite:///{args.study_name}.db'
        print(f"Using SQLite storage: {storage}")
        
        # Optimize SQLite for parallel writes
        import sqlite3
        db_path = storage.replace('sqlite:///', '')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.close()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization
    print(f"\n{'='*60}")
    print(f"🚀 FAST BETTING OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Trials: {args.n_trials}")
    print(f"Parallel workers: {args.n_jobs}")
    print(f"Cached predictions: {len(cached_data['predictions'])}")
    print(f"{'='*60}\n")
    
    study.optimize(
        lambda trial: objective_with_cached_predictions(trial, cached_data),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest Trial:")
    print(f"  Profit: ${study.best_value:.2f}")
    print(f"  ROI: {(study.best_value / 1000 * 100):.2f}%")
    
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print(f"\nTo use these parameters:")
    print(f"""
model = Model(
    min_edge={study.best_params['min_edge']:.4f},
    max_kelly_fraction={study.best_params['max_kelly_fraction']:.4f},
    kelly_multiplier={study.best_params['kelly_multiplier']:.4f},
    normalize_odds={study.best_params['normalize_odds']},
    min_probability={study.best_params['min_probability']:.4f}
)
""")
    
    print(f"\n💡 Tip: Run again without --retrain to use cached predictions (much faster!)")


if __name__ == '__main__':
    main()
