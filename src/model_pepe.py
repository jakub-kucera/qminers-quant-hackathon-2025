


import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict


class Model:
    """
    Hybrid Value Kelly betting strategy with optimized parameters
    """
    
    def __init__(self):
        # Optimized parameters from 200-trial optimization
        self.k_factor = 20.39461606563558
        self.home_advantage = 0.10701183757315036
        self.rolling_window = 14
        self.min_edge = 0.041259553272506584
        self.kelly_fraction = 0.2860406159317642
        self.value_multiplier = 2.9477943166836127
        self.confidence_threshold = 0.5900017227196364
        self.form_adjustment_weight = 0.12081194384929457
        self.streak_adjustment_weight = 0.1961522454451103
        self.initial_rating = 1409.6347280369619
        
        # Team tracking
        self.ratings = {}
        self.team_form = {}
        self.team_streaks = {}
    
    def _calculate_probabilities(self, home_rating: float, away_rating: float) -> List[float]:
        """Calculate win/draw/loss probabilities using Elo"""
        rating_diff = home_rating - away_rating
        prob_home_vs_away = 1.0 / (1.0 + 10 ** (-rating_diff / 400))
        
        # Draw probability based on rating closeness
        draw_factor = 0.25
        rating_closeness = 1.0 - min(abs(rating_diff) / 400, 1.0)
        prob_d = draw_factor * rating_closeness
        
        # Adjust win probabilities
        prob_h = prob_home_vs_away * (1.0 - prob_d)
        prob_a = (1.0 - prob_home_vs_away) * (1.0 - prob_d)
        
        # Normalize
        total = prob_h + prob_a + prob_d
        return [prob_h / total, prob_a / total, prob_d / total]
    
    def _get_adjusted_rating(self, team_id: int, is_home: bool) -> float:
        """Get rating adjusted for home advantage, form, and streak"""
        base_rating = self.ratings.get(team_id, self.initial_rating)
        
        if is_home:
            base_rating += self.home_advantage
        
        # Form adjustment
        form = self.team_form.get(team_id, [])
        if form:
            avg_form = np.mean(form)
            form_adjustment = (avg_form - 0.5) * 400 * self.form_adjustment_weight
            base_rating += form_adjustment
        
        # Streak adjustment
        streak = self.team_streaks.get(team_id, 0)
        streak_adjustment = streak * 20 * self.streak_adjustment_weight
        base_rating += streak_adjustment
        
        return base_rating
    
    def _calculate_confidence(self, probs: List[float], home_id: int, away_id: int) -> float:
        """Calculate confidence score"""
        # Probability separation
        sorted_probs = sorted(probs, reverse=True)
        separation = sorted_probs[0] - sorted_probs[1]
        
        # Entropy-based certainty
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log(3)
        certainty = 1 - (entropy / max_entropy)
        
        # Form consistency
        home_form = self.team_form.get(home_id, [])
        away_form = self.team_form.get(away_id, [])
        
        home_var = np.var(home_form) if len(home_form) > 1 else 0.0
        away_var = np.var(away_form) if len(away_form) > 1 else 0.0
        form_consistency = 1 / (1 + home_var + away_var)
        
        # Weighted combination
        confidence = (
            0.4 * separation +
            0.4 * certainty +
            0.2 * form_consistency
        )
        
        return confidence
    
    def _update_ratings(self, games: pd.DataFrame) -> None:
        """Update Elo ratings and team statistics"""
        for idx, game in games.iterrows():
            home_id = game["HID"]
            away_id = game["AID"]
            home_won = game["H"]
            away_won = game["A"]
            
            # Initialize if needed
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            # Elo update
            home_rating = self.ratings[home_id] + self.home_advantage
            away_rating = self.ratings[away_id]
            
            expected_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400))
            expected_away = 1.0 - expected_home
            
            if home_won:
                actual_home, actual_away = 1.0, 0.0
            elif away_won:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5
            
            self.ratings[home_id] += self.k_factor * (actual_home - expected_home)
            self.ratings[away_id] += self.k_factor * (actual_away - expected_away)
            
            # Update form (rolling window)
            self.team_form[home_id].append(actual_home)
            if len(self.team_form[home_id]) > self.rolling_window:
                self.team_form[home_id].pop(0)
            
            self.team_form[away_id].append(actual_away)
            if len(self.team_form[away_id]) > self.rolling_window:
                self.team_form[away_id].pop(0)
            
            # Update streaks
            if home_won:
                self.team_streaks[home_id] = self.team_streaks[home_id] + 1 if self.team_streaks[home_id] > 0 else 1
                self.team_streaks[away_id] = self.team_streaks[away_id] - 1 if self.team_streaks[away_id] < 0 else -1
            elif away_won:
                self.team_streaks[home_id] = self.team_streaks[home_id] - 1 if self.team_streaks[home_id] < 0 else -1
                self.team_streaks[away_id] = self.team_streaks[away_id] + 1 if self.team_streaks[away_id] > 0 else 1
            else:
                self.team_streaks[home_id] = 0
                self.team_streaks[away_id] = 0
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Place bets using Hybrid Value Kelly strategy"""
        # Update ratings with new results
        if not inc.empty:
            self._update_ratings(inc)
        
        if opps.empty:
            return pd.DataFrame()
        
        # Get constraints
        bankroll = summary["Bankroll"].iloc[0] if not summary.empty else 1000.0
        min_bet = summary.get("Min_bet", pd.Series([10])).iloc[0] if not summary.empty else 10.0
        max_bet = summary.get("Max_bet", pd.Series([np.inf])).iloc[0] if not summary.empty else np.inf
        
        n_games = len(opps)
        bets = np.zeros((n_games, 3))
        
        for i, (_, game) in enumerate(opps.iterrows()):
            home_id = game["HID"]
            away_id = game["AID"]
            
            # Initialize if needed
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            # Get adjusted ratings
            home_rating = self._get_adjusted_rating(home_id, is_home=True)
            away_rating = self._get_adjusted_rating(away_id, is_home=False)
            
            # Calculate probabilities
            probs = self._calculate_probabilities(home_rating, away_rating)
            
            # Get odds
            odds = [game.get("OddsH", 0), game.get("OddsA", 0), game.get("OddsD", 0)]
            if any(o == 0 for o in odds[:2]):
                continue
            
            # Calculate edges
            edges = [probs[j] * odds[j] - 1 for j in range(3)]
            
            # Calculate confidence
            confidence = self._calculate_confidence(probs, home_id, away_id)
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Find best bet
            best_idx = np.argmax(edges)
            best_edge = edges[best_idx]
            
            if best_edge > self.min_edge:
                # Value score
                value_score = best_edge / self.min_edge
                
                # Kelly bet with value multiplier
                kelly_bet = (
                    self.kelly_fraction *
                    best_edge *
                    bankroll *
                    min(self.value_multiplier, value_score) *
                    confidence
                )
                
                # Apply constraints and round to integer
                kelly_bet = np.round(max(min_bet, min(kelly_bet, max_bet, bankroll * 0.1)))
                bets[i, best_idx] = kelly_bet
        
        # Ensure total doesn't exceed bankroll
        total_bet = bets.sum()
        if total_bet > bankroll:
            bets *= (bankroll * 0.95) / total_bet
            bets = np.round(bets)  # Round to integers after scaling
            bets[bets < min_bet] = 0
        
        # Convert to [H, D, A] format
        bets_reordered = np.column_stack([bets[:, 0], bets[:, 2], bets[:, 1]])
        return pd.DataFrame(bets_reordered, index=opps.index, columns=["BetH", "BetD", "BetA"])
