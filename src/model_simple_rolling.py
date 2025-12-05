"""
Simple Rolling Window Model

A minimal model using only rolling window features with binary classification:
- 1 = Home Win
- 0 = Away Win or Draw (combined)

No ensemble, no sub-models, just RollingWindowFeatures → LogisticRegression
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class RollingWindowFeatures:
    """Computes rolling window statistics for teams"""
    
    def __init__(self, window_short=5, window_med=10, window_long=20,
                 elo_k=20, elo_home_advantage=50):
        self.window_short = window_short
        self.window_med = window_med
        self.window_long = window_long
        self.team_history = defaultdict(list)
        
        # ELO rating system
        self.elo_ratings = defaultdict(lambda: 1500)  # Start all teams at 1500
        self.elo_k = elo_k  # K-factor: how quickly ratings change
        self.elo_home_advantage = elo_home_advantage  # ELO points for home ice
        
        # Head-to-head matchup history
        # Key: (home_id, away_id), Value: result of last game (1.0=home won, 0.0=away won)
        self.h2h_history = {}
    
    def add_game(self, current):
        """Add game to history with null-safety"""
        # Check for required fields
        try:
            home_id = current['HID']
            away_id = current['AID']
            home_score = current['HS']
            away_score = current['AS']
            
            # Check for NaN in required fields
            if pd.isna(home_id) or pd.isna(away_id) or pd.isna(home_score) or pd.isna(away_score):
                print(f"⚠️  Skipping game with NaN in required fields")
                return
                
        except KeyError as e:
            print(f"⚠️  Skipping game - missing required column: {e}")
            return
        
        # Helper to safely get values (handles NaN and missing columns)
        def safe_get(key, default=0):
            try:
                val = current.get(key, default)
                return default if pd.isna(val) else val
            except:
                return default
        
        # Store game stats for both teams (using correct column names from problem_info.md)
        home_stats = {
            'goals_for': home_score,
            'goals_against': away_score,
            'sog_for': safe_get('H_SOG', home_score * 3),
            'sog_against': safe_get('A_SOG', away_score * 3),
            'pp_goals': safe_get('H_PPG', 0),
            'pp_opps': max(safe_get('H_PEN', 3), 1),
            'faceoffs_won': safe_get('H_FO', 30),
            'faceoffs_total': max(safe_get('H_FO', 30) + safe_get('A_FO', 30), 1),
            'won': 1 if safe_get('H', False) else 0,
        }
        
        away_stats = {
            'goals_for': away_score,
            'goals_against': home_score,
            'sog_for': safe_get('A_SOG', away_score * 3),
            'sog_against': safe_get('H_SOG', home_score * 3),
            'pp_goals': safe_get('A_PPG', 0),
            'pp_opps': max(safe_get('A_PEN', 3), 1),
            'faceoffs_won': safe_get('A_FO', 30),
            'faceoffs_total': max(safe_get('H_FO', 30) + safe_get('A_FO', 30), 1),
            'won': 1 if safe_get('A', False) else 0,
        }
        
        self.team_history[home_id].append(home_stats)
        self.team_history[away_id].append(away_stats)
        
        # Update ELO ratings
        self._update_elo_ratings(home_id, away_id, home_score, away_score, current)
        
        # Update head-to-head history
        # Store result: 1.0 if home won, 0.0 if away won
        # h2h_result = 1.0 if home_score > away_score else 0.0
        # self.h2h_history[(home_id, away_id)] = h2h_result
        h2h_result = 1.0 if home_score > away_score else 0.0
        self.h2h_history[(home_id, away_id)] = h2h_result
        self.h2h_history[(away_id, home_id)] = 1.0 - h2h_result  # Inverse for reverse matchup
    
    def _update_elo_ratings(self, home_id, away_id, home_score, away_score, game_data):
        """Update ELO ratings based on game outcome"""
        # Get current ratings
        home_elo = self.elo_ratings[home_id]
        away_elo = self.elo_ratings[away_id]
        
        # Calculate expected score (with home advantage)
        expected_home = 1 / (1 + 10**((away_elo - (home_elo + self.elo_home_advantage)) / 400))
        expected_away = 1 - expected_home
        
        # Determine actual score
        # 1.0 for regulation win, 0.75 for OT/SO win, 0.25 for OT/SO loss, 0.0 for regulation loss
        if home_score > away_score:
            # Home won
            if game_data.get('Special', '') in ['OT', 'SO']:
                actual_home = 0.75  # OT/SO win
                actual_away = 0.25  # OT/SO loss (still get a point)
            else:
                actual_home = 1.0  # Regulation win
                actual_away = 0.0  # Regulation loss
        else:
            # Away won
            if game_data.get('Special', '') in ['OT', 'SO']:
                actual_home = 0.25  # OT/SO loss
                actual_away = 0.75  # OT/SO win
            else:
                actual_home = 0.0  # Regulation loss
                actual_away = 1.0  # Regulation win
        
        # Update ratings
        self.elo_ratings[home_id] += self.elo_k * (actual_home - expected_home)
        self.elo_ratings[away_id] += self.elo_k * (actual_away - expected_away)
    
    def _compute_window_stats(self, team_id, window_size):
        """Compute stats for a team over a window"""
        if team_id not in self.team_history:
            print(f"⚠️  Team {team_id} not found in history")
            return None
        
        recent_games = self.team_history[team_id][-window_size:]
        
        if len(recent_games) < min(3, window_size):
            print(f"⚠️  Insufficient data for team {team_id}: {len(recent_games)} games available, need {min(3, window_size)} for window={window_size}")
            return None
        
        # Aggregate stats
        total_goals_for = sum(g['goals_for'] for g in recent_games)
        total_goals_against = sum(g['goals_against'] for g in recent_games)
        total_sog_for = sum(g['sog_for'] for g in recent_games)
        total_sog_against = sum(g['sog_against'] for g in recent_games)
        total_pp_goals = sum(g['pp_goals'] for g in recent_games)
        total_pp_opps = sum(g['pp_opps'] for g in recent_games)
        total_fo_won = sum(g['faceoffs_won'] for g in recent_games)
        total_fo_total = sum(g['faceoffs_total'] for g in recent_games)
        total_wins = sum(g['won'] for g in recent_games)
        
        n_games = len(recent_games)
        
        # Calculate averages and percentages
        avg_goals_for = total_goals_for / n_games
        avg_goals_against = total_goals_against / n_games
        avg_sog_for = total_sog_for / n_games
        avg_sog_against = total_sog_against / n_games
        pp_pct = total_pp_goals / max(total_pp_opps, 0.1)
        fo_win_pct = total_fo_won / max(total_fo_total, 1)
        win_pct = total_wins / n_games
        shooting_pct = total_goals_for / max(total_sog_for, 0.1)
        
        return [
            avg_goals_for,
            avg_goals_against,
            avg_sog_for,
            avg_sog_against,
            pp_pct,
            fo_win_pct,
            win_pct,
            shooting_pct
        ]
    
    def get_input_data(self, home_id, away_id):
        """Get rolling window features for both teams"""
        features = []
        
        for window_size in [self.window_short, self.window_med, self.window_long]:
            home_stats = self._compute_window_stats(home_id, window_size)
            away_stats = self._compute_window_stats(away_id, window_size)
            
            if home_stats is None or away_stats is None:
                # Print warning about insufficient data
                if home_stats is None:
                    home_games = len(self.team_history.get(home_id, []))
                    # print(f"⚠️  Insufficient data for home team {home_id}: {home_games} games available, need {min(3, window_size)} for window={window_size}")
                if away_stats is None:
                    away_games = len(self.team_history.get(away_id, []))
                    # print(f"⚠️  Insufficient data for away team {away_id}: {away_games} games available, need {min(3, window_size)} for window={window_size}")
                return None
            
            # Add differential features (home - away)
            features.extend([
                home_stats[0] - away_stats[0],  # Goal differential
                home_stats[1] - away_stats[1],  # Goals against differential
                home_stats[2] - away_stats[2],  # SOG differential
                home_stats[4] - away_stats[4],  # PP% differential
                home_stats[5] - away_stats[5],  # FO% differential
                home_stats[6] - away_stats[6],  # Win% differential
                home_stats[7] - away_stats[7],  # Shooting% differential
            ])
        
        # Add ELO difference as final feature (22nd feature)
        home_elo = self.elo_ratings.get(home_id, 1500)
        away_elo = self.elo_ratings.get(away_id, 1500)
        elo_diff = (home_elo + self.elo_home_advantage) - away_elo
        features.append(elo_diff / 100)  # Normalize to ~0-5 range
        
        # Add head-to-head result (23rd feature)
        # 1.0 = home won last matchup, 0.0 = away won, 0.5 = no previous matchup
        h2h_result = self.h2h_history.get((home_id, away_id), 0.5)
        features.append(h2h_result)
        
        return features


class Model:
    """
    Simple binary classification model:
    - Uses only RollingWindowFeatures (21 features total)
    - Binary outcome: 1 = Home Win, 0 = Away Win or Draw
    - Single LogisticRegression classifier
    """

#     Best Parameters:
#   min_edge: 0.1374
#   max_kelly_fraction: 0.0980
#   kelly_multiplier: 0.7867
#   normalize_odds: True
    
    def __init__(self,
                 # Rolling window parameters
                 rolling_window_short=5,
                 rolling_window_med=10,
                 rolling_window_long=20,
                 # Training parameters
                 required_games=100,
                 retrain_freq=50,
                 # Betting strategy parameters
                #  min_edge=0.1374,  # Minimum edge to place bet (2%)
                #  max_kelly_fraction=0.0980,  # Maximum Kelly fraction (25%)
                #  kelly_multiplier=0.7867,  # Multiplier on Kelly bet size
                #  normalize_odds=False,
                #  min_probability=0.8,



                # best parames with SOG included. + ~400$
                 min_edge=0.1768,
                 max_kelly_fraction=0.1057,
                 kelly_multiplier=1.7112,
                 normalize_odds=False,
                 min_probability=0.9430


                # # best parames without SOG included. + ~309$
                # min_edge=0.1347,
                # max_kelly_fraction=0.1358,
                # kelly_multiplier=1.3437,
                # normalize_odds=False,
                # min_probability=0.5112

                 ): 
        """
        Initialize simple rolling window model
        
        Args:
            rolling_window_short: Short-term window size (recent form)
            rolling_window_med: Medium-term window size (trends)
            rolling_window_long: Long-term window size (season performance)
            required_games: Minimum games before making predictions
            retrain_freq: How often to retrain the model
        """
        self.required_games = required_games
        self.retrain_freq = retrain_freq
        self.retrain_counter = retrain_freq
        
        # Betting strategy parameters
        self.min_edge = min_edge
        self.max_kelly_fraction = max_kelly_fraction
        self.kelly_multiplier = kelly_multiplier
        self.normalize_odds = normalize_odds
        self.min_probability = min_probability
        
        # Initialize rolling window feature extractor
        self.rolling = RollingWindowFeatures(
            window_short=rolling_window_short,
            window_med=rolling_window_med,
            window_long=rolling_window_long
        )
        
        # Training data storage
        self.training_data = []  # List of [features..., outcome]
        
        # Logistic regression model
        self.model = None
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Accuracy tracking
        self.predictions_made = 0
        self.predictions_correct = 0
        self.accuracy_history = []  # Track accuracy over time
    
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        """
        Make predictions and place bets
        
        Binary prediction: P(Home Win) vs P(Away Win or Draw)
        """
        # Process incremental data (update rolling windows)
        for idx in inc.index:
            current = inc.loc[idx]
            self._process_game(current)
        
        # Initialize bets dataframe
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        
        bets = pd.DataFrame(
            data=np.zeros((len(opps), 3)),
            columns=["BetH", "BetD", "BetA"],
            index=opps.index
        )
        
        # Make predictions if we have enough data
        if len(self.training_data) < self.required_games:
            return bets
        
        # Retrain model if needed
        if self.retrain_counter <= 0:
            self._train_model()
            self.retrain_counter = self.retrain_freq
        
        # Make predictions for each game
        for i in opps.index:
            current = opps.loc[i]
            home_id = current['HID']
            away_id = current['AID']
            
            # Get features
            features = self.rolling.get_input_data(home_id, away_id)
            
            if features is None or self.model is None:
                # Warn if we can't make prediction
                if features is None:
                    print(f"⚠️  Cannot predict game {i} on {current['Date']}: Insufficient rolling window data for teams {home_id} vs {away_id}")
                if self.model is None:
                    print(f"⚠️  Cannot predict game {i}: Model not trained yet")
                continue
            
            # Predict probability of home win
            prob_home_win = self.model.predict_proba(self.scaler.transform([features]))[0][1]
            
            # Get odds
            odds_home = current['OddsH']
            
            # Calculate bookmaker's implied probability
            if self.normalize_odds:
                # Normalized: Remove bookmaker margin
                odds_away = current['OddsA']
                odds_draw = current.get('OddsD', 0)
                
                raw_prob_home = 1.0 / odds_home
                raw_prob_away = 1.0 / odds_away
                raw_prob_draw = 1.0 / odds_draw if odds_draw > 0 else 0
                
                total = raw_prob_home + raw_prob_away + raw_prob_draw
                bookmaker_prob = raw_prob_home / total
            else:
                # Simple: Direct conversion
                bookmaker_prob = 1.0 / odds_home
            
            # Calculate edge
            edge = prob_home_win - bookmaker_prob
            
            # if edge > 0.02:  # Minimum 2% edge
            #     # Kelly criterion
            #     kelly_fraction = edge / (odds_home - 1)
            #     kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            # Better logic
            if edge > self.min_edge and prob_home_win > self.min_probability:
                # Kelly criterion
                kelly_fraction = edge / (odds_home - 1)
                kelly_fraction = kelly_fraction * self.kelly_multiplier  # Apply multiplier
                kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))  # Apply cap

                
                bet_amount = kelly_fraction * bankroll
                bet_amount = max(min_bet, min(bet_amount, max_bet))
                bet_amount = np.round(bet_amount)
            
                bets.at[i, 'BetH'] = bet_amount
        
        return bets
    
    def _process_game(self, game):
        """Process a completed game and track accuracy"""
        home_id = game['HID']
        away_id = game['AID']
        
        # Get features before updating
        features = self.rolling.get_input_data(home_id, away_id)
        
        # Determine binary outcome: 1 = Home Win, 0 = Away Win or Draw
        outcome = 1 if game['H'] else 0
        
        # Track accuracy if we have a trained model
        if features is not None and self.model is not None:
            prediction = self.model.predict(self.scaler.transform([features]))[0]
            self.predictions_made += 1
            if prediction == outcome:
                self.predictions_correct += 1
            
            # Log accuracy every 50 games
            if self.predictions_made % 50 == 0:
                accuracy = self.predictions_correct / self.predictions_made
                self.accuracy_history.append(accuracy)
                print(f"📊 Model Accuracy: {accuracy:.2%} ({self.predictions_correct}/{self.predictions_made})")
        
        # Add to training data if features available
        if features is not None:
            self.training_data.append(features + [outcome])
            self.retrain_counter -= 1
        
        # Update rolling windows
        self.rolling.add_game(game)
    
    def _train_model(self):
        """Train the logistic regression model"""
        if len(self.training_data) < self.required_games:
            return
        
        # Convert to numpy array
        data = np.array(self.training_data)
        X = data[:, :-1]  # Features
        y = data[:, -1]   # Binary outcome
        
        # Fit scaler on training data and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply exponential decay weighting (recent games more important)
        sample_weights = np.exp(-0.0003 * np.arange(len(data)))[::-1]
        
        # Train binary logistic regression on scaled features
        self.model = LogisticRegression(
            max_iter=10000,
            solver='lbfgs',
            random_state=42
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
