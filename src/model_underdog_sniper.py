"""
Underdog Sniper Model

A highly selective strategy targeting high-EV underdogs (Odds >= 3.0).
Features:
- Elo ratings
- Recent form (Rolling stats)
- Poisson expectations
- Exhaustion features

Logic:
1. Predict probabilities using Logistic Regression.
2. Calibrate probabilities using Isotonic Regression (on past predictions).
3. Filter for Underdogs (Odds >= 3.0).
4. Check EV >= 20% using calibrated probability.
5. Place fixed $250 bets.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from model import Model as BaseModel
from model import NateSilverElo, RollingWindowFeatures, PoissonGoalModel, Exhaustion, HockeyFourFactor

class Model(BaseModel):
    def __init__(self, 
                 calibration_window=1000, 
                 min_calibration_samples=100,
                 fixed_bet_size=250.0,
                 ev_threshold=0.20,
                 min_odds=3.0):
        
        # Initialize features (using subsets of what's available in model.py)
        # We'll use a robust set of features
        self.model_list = [
            NateSilverElo(k_base=20, home_advantage=40, margin_cap=5),
            RollingWindowFeatures(window_short=3, window_med=11, window_long=22),
            PoissonGoalModel(regularization=3.5, daily_decay=0.995),
            Exhaustion(b2b_penalty=1.6, one_day_penalty=0.8),
            HockeyFourFactor(lr_required_n=400, daily_decay=0.99)
        ]
        
        # Classifier
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        self.is_fitted = False
        
        # Training data buffer
        self.X_train = []
        self.y_train = []
        
        # Calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        self.calibration_data = {
            'prob': [],   # Raw predicted probabilities
            'outcome': [] # Binary outcome (1 if event happened, 0 otherwise)
        }
        self.calibration_window = calibration_window
        self.min_calibration_samples = min_calibration_samples
        self.is_calibrated = False
        
        # Strategy params
        self.fixed_bet_size = fixed_bet_size
        self.ev_threshold = ev_threshold
        self.min_odds = min_odds
        
        # State
        self.past_pred_prob = {} # Store raw probs to pair with outcomes later

    def _get_input_features(self, home_id, away_id, season, date):
        """Collect features from all sub-models"""
        input_data = []
        for model in self.model_list:
            if isinstance(model, NateSilverElo):
                data = model.get_input_data(home_id, away_id, season)
            elif isinstance(model, (RollingWindowFeatures, PoissonGoalModel, Exhaustion, HockeyFourFactor)):
                data = model.get_input_data(home_id, away_id, date)
            else:
                data = [] # Should not happen based on init
                
            if data is None:
                return None
            input_data.extend(data)
        return input_data

    def _update_models_and_data(self, inc):
        """Process incremental updates"""
        for idx, row in inc.iterrows():
            # Update Sub-models
            for model in self.model_list:
                model.add_game(row)
            
            # Prepare outcome labels (3-class for LR)
            if row['H']: outcome = 2    # Home
            elif row['A']: outcome = 0  # Away
            else: outcome = 1           # Draw
            
            # Add to main classifier training data
            # (Note: In a real incremental setting we'd only add if we had features *before* the game)
            # Here we regenerate features for the game time to ensure consistency
            feats = self._get_input_features(row['HID'], row['AID'], row['Season'], row['Date'])
            if feats is not None:
                self.X_train.append(feats)
                self.y_train.append(outcome)
            
            # Calibration Update
            # Get the probability we predicted for this game (if we did)
            if idx in self.past_pred_prob:
                raw_probs = self.past_pred_prob[idx] # [P(A), P(D), P(H)]
                
                # We calibrate binary outcomes regarding "Win" for Home and Away
                # Add (P(Home), DidHomeWin)
                self.calibration_data['prob'].append(raw_probs[2])
                self.calibration_data['outcome'].append(1 if row['H'] else 0)
                
                # Add (P(Away), DidAwayWin)
                self.calibration_data['prob'].append(raw_probs[0])
                self.calibration_data['outcome'].append(1 if row['A'] else 0)
                
                # Cleanup old calibration data
                if len(self.calibration_data['prob']) > self.calibration_window * 2:
                    start = len(self.calibration_data['prob']) - self.calibration_window * 2
                    self.calibration_data['prob'] = self.calibration_data['prob'][start:]
                    self.calibration_data['outcome'] = self.calibration_data['outcome'][start:]

    def _retrain_if_needed(self):
        """Retrain classifier and calibrator"""
        # Train Classifier
        if len(self.y_train) >= 500: # Min samples to start
            # Retrain every X games or just simple check? 
            # For efficiency let's check size, but here we assume 'inc' comes daily. 
            # We'll just retrain if we have enough data. 
            # Optim: Retrain every 100 new samples?
            if len(self.y_train) % 100 == 0:
                self.classifier.fit(self.X_train, self.y_train)
                self.is_fitted = True
        
        # Train Calibrator
        if len(self.calibration_data['prob']) >= self.min_calibration_samples:
            if len(self.calibration_data['prob']) % 100 == 0: # Periodic update
                self.calibrator.fit(self.calibration_data['prob'], self.calibration_data['outcome'])
                self.is_calibrated = True

    def place_bets(self, summary, opps, inc):
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        
        # 1. Update data
        self._update_models_and_data(inc)
        
        # 2. Retrain
        self._retrain_if_needed()
        
        bets = pd.DataFrame(
            data=np.zeros((len(opps), 3)),
            columns=["BetH", "BetA", "BetD"],
            index=opps.index
        )
        
        if not self.is_fitted:
            return bets

        # 3. Predict & Bet
        for idx, row in opps.iterrows():
            feats = self._get_input_features(row['HID'], row['AID'], row['Season'], row['Date'])
            if feats is None:
                continue
                
            # Raw Probabilities: [Away, Draw, Home] (classes 0, 1, 2 sorted)
            try:
                probs = self.classifier.predict_proba([feats])[0]
            except Exception:
                continue
                
            # Store for calibration later
            self.past_pred_prob[idx] = probs
            
            p_away_raw = probs[0]
            p_home_raw = probs[2]
            
            # Calibration (if available)
            if self.is_calibrated:
                p_away = self.calibrator.transform([p_away_raw])[0]
                p_home = self.calibrator.transform([p_home_raw])[0]
                # Fallback ensures non-negative (isotonic clips but just in case)
                p_away = max(0.001, p_away)
                p_home = max(0.001, p_home)
            else:
                p_away = p_away_raw
                p_home = p_home_raw
            
            # Betting Logic
            # Check Underdog status (Odds >= 3.0)
            
            # Home Underdog?
            if row['OddsH'] >= self.min_odds:
                ev_home = (row['OddsH'] * p_home) - 1.0
                if ev_home >= self.ev_threshold:
                    # Place Bet
                    amount = min(self.fixed_bet_size, max_bet, bankroll)
                    if amount >= min_bet:
                        bets.at[idx, 'BetH'] = amount
                        bankroll -= amount # Decrement locally for this simplified logic
            
            # Away Underdog?
            if row['OddsA'] >= self.min_odds:
                ev_away = (row['OddsA'] * p_away) - 1.0
                if ev_away >= self.ev_threshold:
                    # Place Bet
                    amount = min(self.fixed_bet_size, max_bet, bankroll)
                    if amount >= min_bet:
                        bets.at[idx, 'BetA'] = amount
                        bankroll -= amount
                        
        return bets
