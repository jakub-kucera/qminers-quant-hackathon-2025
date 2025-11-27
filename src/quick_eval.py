"""
Quick model evaluation script for testing accuracy without betting.
Useful for rapid iteration during development.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

# sys.path.insert(0, '.')
from model import Model
from environment import Environment


def evaluate_predictions(games, model, start_date, end_date):
    """
    Evaluate model prediction accuracy without betting constraints
    Focuses on pure ensemble prediction accuracy for all games
    
    Args:
        games: DataFrame with game data
        model: Model instance
        start_date: Start date for evaluation
        end_date: End date for evaluation
    
    Returns:
        Dictionary with metrics
    """
    env = Environment(
        games.copy(),
        model,
        start_date=start_date,
        end_date=end_date,
        init_bankroll=1000,
        min_bet=5,
        max_bet=100
    )
    
    predictions = []
    dates_processed = 0
    
    print(f"\nEvaluating from {start_date.date()} to {end_date.date()}...")
    
    for date in pd.date_range(start_date, end_date):
        inc = env._next_date(date)
        opps = env._get_options(date)
        
        if opps.empty and inc.empty:
            continue
        
        dates_processed += 1
        if dates_processed % 100 == 0:
            print(f"  Processed {dates_processed} dates...")
        
        # Process incremental data to update model state
        for idx in inc.index:
            current = inc.loc[idx]
            model._game_increment(idx, current)
        
        # Make predictions for ALL games (not dependent on betting)
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
                        from sklearn.linear_model import LogisticRegression
                        np_array = np.array(model.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(model.past_pred)))
                        model.ensamble = LogisticRegression(max_iter=10000)
                        model.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
                    
                    # Get prediction from ensemble
                    pred_home = model.ensamble.predict_proba(np.array([input_arr]))[0, 1]
                    
                    game = games.loc[idx]
                    
                    # Calculate market probability
                    odds_home = game['OddsH']
                    odds_away = game['OddsA']
                    odds_draw = game['OddsD']
                    
                    if odds_home > 0 and odds_away > 0:
                        overround = 1/odds_home + 1/odds_away + (1/odds_draw if odds_draw > 0 else 0)
                        market_prob = (1/odds_home) / overround
                    else:
                        market_prob = 0.5
                    
                    predictions.append({
                        'date': game['Date'],
                        'pred_home': pred_home,
                        'pred_away': 1 - pred_home,  # Simplified 2-way for now
                        'market_home': market_prob,
                        'market_away': 1 - market_prob,
                        'actual_home': int(game['H']),
                        'actual_away': int(game['A']),
                        'actual_draw': int(game['D']),
                        'odds_home': odds_home,
                        'odds_away': odds_away,
                        'odds_draw': odds_draw
                    })
    
    return calculate_metrics(predictions)


def calculate_metrics(predictions):
    """Calculate comprehensive metrics"""
    if not predictions:
        return {'error': 'No predictions'}
    
    n = len(predictions)
    
    # Binary accuracy (home win vs not home win)
    my_correct = sum(1 for p in predictions if (p['pred_home'] > 0.5) == p['actual_home'])
    mkt_correct = sum(1 for p in predictions if (p['market_home'] > 0.5) == p['actual_home'])
    
    # Log loss
    my_log_loss = 0.0
    mkt_log_loss = 0.0
    for p in predictions:
        my_p = max(0.001, min(0.999, p['pred_home']))
        mkt_p = max(0.001, min(0.999, p['market_home']))
        actual = p['actual_home']
        
        my_log_loss += -(actual * np.log(my_p) + (1-actual) * np.log(1-my_p))
        mkt_log_loss += -(actual * np.log(mkt_p) + (1-actual) * np.log(1-mkt_p))
    
    # Brier score (MSE)
    my_brier = sum((p['pred_home'] - p['actual_home'])**2 for p in predictions)
    mkt_brier = sum((p['market_home'] - p['actual_home'])**2 for p in predictions)
    
    # Calibration error
    n_bins = 10
    my_bins = defaultdict(lambda: {'total': 0, 'positive': 0})
    mkt_bins = defaultdict(lambda: {'total': 0, 'positive': 0})
    
    for p in predictions:
        my_bin = min(int(p['pred_home'] * n_bins), n_bins - 1)
        mkt_bin = min(int(p['market_home'] * n_bins), n_bins - 1)
        
        my_bins[my_bin]['total'] += 1
        my_bins[my_bin]['positive'] += p['actual_home']
        
        mkt_bins[mkt_bin]['total'] += 1
        mkt_bins[mkt_bin]['positive'] += p['actual_home']
    
    my_ece = 0.0
    mkt_ece = 0.0
    for i in range(n_bins):
        if my_bins[i]['total'] > 0:
            bin_accuracy = my_bins[i]['positive'] / my_bins[i]['total']
            bin_confidence = (i + 0.5) / n_bins
            my_ece += abs(bin_accuracy - bin_confidence) * my_bins[i]['total'] / n
        
        if mkt_bins[i]['total'] > 0:
            bin_accuracy = mkt_bins[i]['positive'] / mkt_bins[i]['total']
            bin_confidence = (i + 0.5) / n_bins
            mkt_ece += abs(bin_accuracy - bin_confidence) * mkt_bins[i]['total'] / n
    
    # ROI if betting on all predictions where we're confident
    threshold = 0.55
    confident_bets = [p for p in predictions if p['pred_home'] > threshold or p['pred_home'] < (1-threshold)]
    
    roi = 0.0
    if confident_bets:
        for p in confident_bets:
            if p['pred_home'] > threshold:  # Bet on home
                roi += (p['actual_home'] * p['odds_home']) - 1
            else:  # Bet on away
                roi += (p['actual_away'] * p['odds_away']) - 1
    
    return {
        'n_predictions': n,
        'my_accuracy': my_correct / n,
        'market_accuracy': mkt_correct / n,
        'my_log_loss': my_log_loss / n,
        'market_log_loss': mkt_log_loss / n,
        'my_brier_score': my_brier / n,
        'market_brier_score': mkt_brier / n,
        'my_calibration_error': my_ece,
        'market_calibration_error': mkt_ece,
        'confident_bets': len(confident_bets),
        'confident_roi': roi / len(confident_bets) if confident_bets else 0.0,
        'total_roi': roi
    }


def print_metrics(metrics, title="METRICS"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print(f"\nPredictions: {metrics['n_predictions']}")
    
    print(f"\nACCURACY:")
    print(f"  My Model:    {metrics['my_accuracy']:.4f}")
    print(f"  Market:      {metrics['market_accuracy']:.4f}")
    print(f"  Improvement: {(metrics['my_accuracy'] - metrics['market_accuracy']):.4f}")
    
    print(f"\nLOG LOSS (lower is better):")
    print(f"  My Model:    {metrics['my_log_loss']:.4f}")
    print(f"  Market:      {metrics['market_log_loss']:.4f}")
    print(f"  Improvement: {(metrics['market_log_loss'] - metrics['my_log_loss']):.4f}")
    
    print(f"\nBRIER SCORE (lower is better):")
    print(f"  My Model:    {metrics['my_brier_score']:.4f}")
    print(f"  Market:      {metrics['market_brier_score']:.4f}")
    print(f"  Improvement: {(metrics['market_brier_score'] - metrics['my_brier_score']):.4f}")
    
    print(f"\nCALIBRATION ERROR (lower is better):")
    print(f"  My Model:    {metrics['my_calibration_error']:.4f}")
    print(f"  Market:      {metrics['market_calibration_error']:.4f}")
    
    print(f"\nROI (hypothetical betting):")
    print(f"  Confident bets: {metrics['confident_bets']}")
    print(f"  ROI per bet:    {metrics['confident_roi']:.4f}")
    print(f"  Total ROI:      {metrics['total_roi']:.2f}")


def main():
    """Quick evaluation on recent data"""
    print("Loading data...")
    games = pd.read_csv('../data/games.csv', index_col=0, parse_dates=["Date", "Open"])
    print(f"Loaded {len(games)} games")
    
    # Evaluate on last 2 years
    max_date = games['Date'].max()
    eval_start = max_date - pd.Timedelta(days=730)  # 2 years
    
    print(f"\nCreating model...")
    model = Model()
    
    print(f"\nEvaluating model...")
    metrics = evaluate_predictions(games, model, eval_start, max_date)
    
    print_metrics(metrics, "MODEL EVALUATION (Last 2 Years)")
    
    # Also evaluate on different time periods
    print(f"\n\n{'='*60}")
    print("PERIOD ANALYSIS")
    print(f"{'='*60}")
    
    periods = [
        ("Last 6 months", 180),
        ("Last 1 year", 365),
        ("Last 2 years", 730),
    ]
    
    for period_name, days in periods:
        period_start = max_date - pd.Timedelta(days=days)
        model_period = Model()
        metrics_period = evaluate_predictions(games, model_period, period_start, max_date)
        
        print(f"\n{period_name}:")
        print(f"  Predictions: {metrics_period['n_predictions']}")
        print(f"  Accuracy:    {metrics_period['my_accuracy']:.4f} (market: {metrics_period['market_accuracy']:.4f})")
        print(f"  Log Loss:    {metrics_period['my_log_loss']:.4f} (market: {metrics_period['market_log_loss']:.4f})")
        print(f"  Brier:       {metrics_period['my_brier_score']:.4f} (market: {metrics_period['market_brier_score']:.4f})")


if __name__ == '__main__':
    main()
