"""
Bookie Rookie Model: Ensemble Predictions + Pepe's Betting Strategy for Bookie Rookie Dataset

This model is based on model_hybrid.py but adapted for the bookie_rookie_games.csv dataset.
It combines:
- Sophisticated ensemble prediction model (7 sub-models with multinomial logistic regression)
- Pepe's optimized betting strategy (confidence scoring, value-based Kelly, single-best-outcome)
- No draw betting (draws cannot happen in this context)
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from model import Model as BaseModel


class Model(BaseModel):
    """
    Bookie Rookie model using ensemble predictions with Pepe's betting strategy
    """
    
    def __init__(self, 
                 # Prediction model parameters - optimized via optimize_bookie_rookie.py
                 # Trial 194: +$30,305.31 profit (3,031% ROI!) - improved from trial 191
                 ensamble_required_n=700,
                 ensamble_retrain_freq=150,
                 pythagorean_power=1.031890264397898,
                 pythagorean_regularization=30,
                 pythagorean_decay=0.9893202028961737,
                 four_factor_required_n=300,
                 four_factor_decay=0.9917848635848201,
                 gradient_descent_epochs=15,
                 gradient_descent_min_games=500,
                 exhaustion_b2b_penalty=1.6526715430456378,
                 exhaustion_road_penalty=0.3095261468561664,
                 elo_k_base=15,
                 elo_home_advantage=55,
                 elo_margin_cap=7,
                 poisson_regularization=3.708116512060344,
                 poisson_decay=0.992045263584752,
                 rolling_window_short=3,
                 rolling_window_med=11,
                 rolling_window_long=24,
                 # Betting strategy parameters - optimized
                 pepe_min_edge=0.07993205778464312,
                 pepe_kelly_fraction=0.3084802286209483,
                 pepe_value_multiplier=1.7611869443210917,
                 pepe_confidence_threshold=0.4502471850554413,
                 pepe_max_bet_fraction=0.24799482880841384,
                 pepe_max_total_exposure=0.8991473285385401,
                 # Confidence calculation weights - form-focused approach
                 pepe_confidence_weight_separation=0.11548032865322859,
                 pepe_confidence_weight_certainty=0.2437031965069546,
                 pepe_confidence_weight_form=0.6408164748398168):
        """
        Initialize bookie rookie model with prediction and betting parameters.
        
        Args:
            [Prediction parameters - optimized]
            pepe_min_edge: Minimum edge required to place bet (8.0%)
            pepe_kelly_fraction: Kelly criterion fraction (30.8%)
            pepe_value_multiplier: Maximum value multiplier (1.76x)
            pepe_confidence_threshold: Minimum confidence to bet (45.0%)
            pepe_max_bet_fraction: Maximum bet as fraction of bankroll (24.8%)
            pepe_max_total_exposure: Maximum total exposure across all bets (89.9%)
            pepe_confidence_weight_separation: Weight for probability separation (11.5%)
            pepe_confidence_weight_certainty: Weight for entropy-based certainty (24.4%)
            pepe_confidence_weight_form: Weight for form consistency (64.1%)
        """
        # Initialize base model with prediction parameters
        super().__init__(
            ensamble_required_n=ensamble_required_n,
            ensamble_retrain_freq=ensamble_retrain_freq,
            pythagorean_power=pythagorean_power,
            pythagorean_regularization=pythagorean_regularization,
            pythagorean_decay=pythagorean_decay,
            four_factor_required_n=four_factor_required_n,
            four_factor_decay=four_factor_decay,
            gradient_descent_epochs=gradient_descent_epochs,
            gradient_descent_min_games=gradient_descent_min_games,
            exhaustion_b2b_penalty=exhaustion_b2b_penalty,
            exhaustion_road_penalty=exhaustion_road_penalty,
            elo_k_base=elo_k_base,
            elo_home_advantage=elo_home_advantage,
            elo_margin_cap=elo_margin_cap,
            poisson_regularization=poisson_regularization,
            poisson_decay=poisson_decay,
            rolling_window_short=rolling_window_short,
            rolling_window_med=rolling_window_med,
            rolling_window_long=rolling_window_long,
            # Disable base betting strategy parameters
            bet_threshold=0.0,
            bet_size_base=0.0,
            bet_kelly_fraction=0.0,
            bet_min_confidence=0.0,
            bet_max_exposure=1.0,
            bet_size_max_multiplier=1.0,
            use_overround_adjustment=False,
            confidence_scaling=False
        )
        
        # Pepe's betting strategy parameters
        self.pepe_min_edge = pepe_min_edge
        self.pepe_kelly_fraction = pepe_kelly_fraction
        self.pepe_value_multiplier = pepe_value_multiplier
        self.pepe_confidence_threshold = pepe_confidence_threshold
        self.pepe_max_bet_fraction = pepe_max_bet_fraction
        self.pepe_max_total_exposure = pepe_max_total_exposure
        
        # Confidence calculation weights - normalize to ensure they sum to 1.0
        weight_sum = (pepe_confidence_weight_separation + 
                     pepe_confidence_weight_certainty + 
                     pepe_confidence_weight_form)
        self.pepe_confidence_weight_separation = pepe_confidence_weight_separation / weight_sum
        self.pepe_confidence_weight_certainty = pepe_confidence_weight_certainty / weight_sum
        self.pepe_confidence_weight_form = pepe_confidence_weight_form / weight_sum
        
        # Track recent results for form consistency calculation
        self.team_recent_results = defaultdict(list)
        self.rolling_window = 14
    
    def _calculate_confidence_pepe(self, probs, home_id, away_id):
        """
        Calculate confidence using Pepe's multi-factor approach:
        - Probability separation (difference between top 2 probabilities)
        - Entropy-based certainty (how "sure" the model is)
        - Form consistency (based on variance of recent results)
        """
        # 1. Probability separation (difference between top 2 probabilities)
        sorted_probs = sorted(probs, reverse=True)
        separation = sorted_probs[0] - sorted_probs[1]
        
        # 2. Entropy-based certainty
        epsilon = 1e-10
        entropy = -sum(p * np.log(p + epsilon) for p in probs if p > 0)
        max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
        certainty = 1.0 - (entropy / max_entropy)
        
        # 3. Form consistency (based on variance of recent results)
        home_form = self.team_recent_results.get(home_id, [])
        away_form = self.team_recent_results.get(away_id, [])
        
        home_var = np.var(home_form) if len(home_form) > 1 else 0.0
        away_var = np.var(away_form) if len(away_form) > 1 else 0.0
        form_consistency = 1.0 / (1.0 + home_var + away_var)
        
        # Weighted combination (using parametrized weights)
        confidence = (
            self.pepe_confidence_weight_separation * separation +
            self.pepe_confidence_weight_certainty * certainty +
            self.pepe_confidence_weight_form * form_consistency
        )
        
        return confidence
    
    def _update_team_form(self, game):
        """Update rolling window of team results for form consistency"""
        home_id = game['HID']
        away_id = game['AID']
        
        # Determine outcomes (1.0 = win, 0.5 = draw, 0.0 = loss)
        if game['H']:
            home_result, away_result = 1.0, 0.0
        elif game['A']:
            home_result, away_result = 0.0, 1.0
        else:  # Draw
            home_result, away_result = 0.5, 0.5
        
        # Update form with rolling window
        self.team_recent_results[home_id].append(home_result)
        if len(self.team_recent_results[home_id]) > self.rolling_window:
            self.team_recent_results[home_id].pop(0)
        
        self.team_recent_results[away_id].append(away_result)
        if len(self.team_recent_results[away_id]) > self.rolling_window:
            self.team_recent_results[away_id].pop(0)
    
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        """
        Place bets using ensemble predictions with Pepe's betting strategy
        """
        # Process incremental data using parent's method
        for idx in inc.index:
            current = inc.loc[idx]
            self._game_increment(idx, current)
            # Also update form for Pepe's confidence calculation
            self._update_team_form(current)
        
        # Initialize bets dataframe
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        
        bets = pd.DataFrame(
            data=np.zeros((len(opps), 3)),
            columns=["BetH", "BetD", "BetA"],
            index=opps.index
        )
        
        # Track total exposure
        total_bet_amount = 0.0
        
        # Make predictions and place bets
        for i in opps.index:
            current = opps.loc[i]
            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']
            
            # Get predictions from ensemble model
            if len(self.past_pred) >= self.ensamble_required_n:
                input_arr = self._get_input_features(home_id, away_id, season, date)
                
                if input_arr is not None:
                    # Retrain ensemble if needed
                    if self.ensamble_retrain <= 0:
                        self.ensamble_retrain = self.ensamble_retrain_freq
                        np_array = np.array(self.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                        from sklearn.linear_model import LogisticRegression
                        self.ensamble = LogisticRegression(
                            max_iter=10000,
                            solver='lbfgs'
                        )
                        self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
                    
                    # Get 3-way predictions: [P(Away), P(Draw), P(Home)]
                    probs = self.ensamble.predict_proba(np.array([input_arr]))[0]
                    prob_away, prob_draw, prob_home = probs[0], probs[1], probs[2]
                    self.prediction_map[i] = probs
                    
                    # Get betting odds
                    odds_home = current['OddsH']
                    odds_away = current['OddsA']
                    odds_draw = current['OddsD']
                    
                    # Skip if any odds are invalid
                    if any(odds <= 1.0 for odds in [odds_home, odds_away, odds_draw]):
                        continue
                    
                    # Organize probabilities and odds as [Home, Draw, Away]
                    probs_ordered = [prob_home, prob_draw, prob_away]
                    odds_ordered = [odds_home, odds_draw, odds_away]
                    
                    # Calculate edges for Home and Away only (draws cannot happen)
                    edge_home = prob_home * odds_home - 1
                    edge_away = prob_away * odds_away - 1
                    
                    # Calculate confidence using Pepe's method
                    confidence = self._calculate_confidence_pepe(probs_ordered, home_id, away_id)
                    
                    # Check confidence threshold
                    if confidence < self.pepe_confidence_threshold:
                        continue
                    
                    # Find best bet between Home and Away only
                    if edge_home > edge_away:
                        best_idx = 0  # Home
                        best_edge = edge_home
                    else:
                        best_idx = 2  # Away
                        best_edge = edge_away
                    
                    # Check minimum edge
                    if best_edge > self.pepe_min_edge:
                        # Calculate value score (Pepe's approach)
                        value_score = best_edge / self.pepe_min_edge
                        
                        # Kelly bet with value multiplier and confidence scaling
                        kelly_bet = (
                            self.pepe_kelly_fraction *
                            best_edge *
                            bankroll *
                            min(self.pepe_value_multiplier, value_score) *
                            confidence
                        )
                        
                        # Apply constraints
                        kelly_bet = max(min_bet, min(
                            kelly_bet,
                            max_bet,
                            bankroll * self.pepe_max_bet_fraction
                        ))
                        
                        # Round to integer
                        kelly_bet = np.round(kelly_bet)
                        
                        # Check if we can afford it
                        if total_bet_amount + kelly_bet <= bankroll * self.pepe_max_total_exposure:
                            # Place bet on best outcome (Home or Away only)
                            if best_idx == 0:  # Home
                                bets.at[i, 'BetH'] = kelly_bet
                            else:  # Away (best_idx == 2)
                                bets.at[i, 'BetA'] = kelly_bet
                            
                            total_bet_amount += kelly_bet
        
        # Final check: ensure total doesn't exceed max exposure
        total_bet = bets.sum().sum()
        if total_bet > bankroll * self.pepe_max_total_exposure:
            scale_factor = (bankroll * self.pepe_max_total_exposure) / total_bet
            bets = bets * scale_factor
            bets = np.round(bets)
            # Zero out bets below minimum
            bets[bets < min_bet] = 0
        
        return bets
