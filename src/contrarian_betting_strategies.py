"""
Contrarian Betting Strategies - Betting against favorites when confidence is low

Mathematical Foundation:
1. Favorite-Longshot Bias (Thaler & Ziemba, 1988): Favorites are often overbet, 
   underdogs underbet, creating value in contrarian positions
2. Low-Confidence Contrarian: When model confidence is low, bet against the 
   market favorite or on the draw
3. Kelly Criterion for contrarian bets with adjusted edge estimates
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from scipy.stats import entropy

from models import EloModel


class ContrarianBettingStrategy(EloModel):
    """
    Contrarian strategy that bets against favorites or on draws when confidence is low
    
    Approach:
    - High confidence (>threshold): Standard Kelly betting on favorite
    - Low confidence (<threshold): Contrarian bets on underdog/draw
    - Uses Favorite-Longshot Bias mathematics
    - Multiple threshold levels for progressive contrarian action
    
    Mathematical Basis:
    - Favorite-Longshot Bias: E(underdog) > E(favorite) in uncertain markets
    - Kelly with adjusted probabilities for contrarian positions
    - Confidence-weighted bet sizing
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        high_confidence_threshold: float = 0.70,  # Above: bet favorite
        low_confidence_threshold: float = 0.45,   # Below: bet contrarian
        min_edge: float = 0.05,
        kelly_fraction: float = 0.20,
        contrarian_kelly_fraction: float = 0.10,  # More conservative for contrarian
        draw_bias_multiplier: float = 1.5,  # Extra weight for draw bets
        underdog_bias_multiplier: float = 1.3,  # Extra weight for underdog bets
        form_adjustment_weight: float = 0.1,
        streak_adjustment_weight: float = 0.05,
        initial_rating: float = 1500.0,
    ):
        super().__init__(
            k_factor=k_factor,
            home_advantage=home_advantage,
            rolling_window=rolling_window,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            form_adjustment_weight=form_adjustment_weight,
            streak_adjustment_weight=streak_adjustment_weight,
            initial_rating=initial_rating,
        )
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.contrarian_kelly_fraction = contrarian_kelly_fraction
        self.draw_bias_multiplier = draw_bias_multiplier
        self.underdog_bias_multiplier = underdog_bias_multiplier
    
    def _get_probabilities(self, home_id: int, away_id: int) -> List[float]:
        """
        Get win probabilities for home, away, and draw.
        Returns: [prob_home, prob_away, prob_draw]
        """
        home_rating = self._get_adjusted_rating(home_id, is_home=True)
        away_rating = self._get_adjusted_rating(away_id, is_home=False)
        prob_h, prob_a, prob_d = self._calculate_probabilities(home_rating, away_rating)
        return [prob_h, prob_a, prob_d]
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets using contrarian approach based on confidence.
        Overrides EloModel.place_bets to implement contrarian logic.
        """
        # Update ratings with new game results
        if not inc.empty:
            self._update_ratings(inc)
        
        # Initialize empty bets
        bets = pd.DataFrame(
            0.0,
            index=opps.index,
            columns=["BetH", "BetA", "BetD"]
        )
        
        if opps.empty:
            return bets
        
        # Get betting constraints
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        
        # Calculate bets for each opportunity
        for idx, row in opps.iterrows():
            home_id = row["HID"]
            away_id = row["AID"]
            odds_h = row["OddsH"]
            odds_a = row["OddsA"]
            odds_d = row["OddsD"]
            
            # Skip if no valid odds
            if odds_h == 0 or odds_a == 0 or odds_d == 0:
                continue
            
            # Ensure teams have ratings
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            # Get probabilities
            probs = self._get_probabilities(home_id, away_id)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(probs, home_id, away_id)
            
            # Calculate edges
            edges = [
                probs[0] * odds_h - 1,  # home edge
                probs[2] * odds_d - 1,  # draw edge  
                probs[1] * odds_a - 1   # away edge
            ]
            
            # Determine betting strategy based on confidence
            if confidence >= self.high_confidence_threshold:
                # HIGH CONFIDENCE: Standard Kelly on favorite
                bet_amount = self._bet_on_favorite(probs, [odds_h, odds_a, odds_d], edges, bankroll)
                if bet_amount > min_bet:
                    favorite_idx = np.argmax(probs)
                    if favorite_idx == 0:
                        bets.at[idx, "BetH"] = bet_amount
                    elif favorite_idx == 1:
                        bets.at[idx, "BetA"] = bet_amount
                    else:
                        bets.at[idx, "BetD"] = bet_amount
            
            elif confidence <= self.low_confidence_threshold:
                # LOW CONFIDENCE: Contrarian betting (underdog or draw)
                bet_result = self._bet_contrarian(probs, [odds_h, odds_a, odds_d], edges, bankroll, confidence)
                if bet_result is not None:
                    bet_idx, bet_amount = bet_result
                    if bet_amount > min_bet:
                        if bet_idx == 0:
                            bets.at[idx, "BetH"] = bet_amount
                        elif bet_idx == 1:
                            bets.at[idx, "BetD"] = bet_amount
                        else:
                            bets.at[idx, "BetA"] = bet_amount
            
            # MEDIUM CONFIDENCE: No bet (uncertain)
        
        # Scale bets if total exceeds bankroll
        total_bet = bets[["BetH", "BetA", "BetD"]].sum().sum()
        if total_bet > bankroll:
            scale_factor = (bankroll * 0.95) / total_bet
            bets[["BetH", "BetA", "BetD"]] *= scale_factor
            # Re-apply min_bet filter
            bets.loc[bets["BetH"] < min_bet, "BetH"] = 0
            bets.loc[bets["BetA"] < min_bet, "BetA"] = 0
            bets.loc[bets["BetD"] < min_bet, "BetD"] = 0
        
        return bets
    
    def _bet_on_favorite(
        self, probs: List[float], odds: List[float], edges: List[float], bankroll: float
    ) -> float:
        """
        Standard Kelly bet on the favorite outcome
        """
        favorite_idx = np.argmax(probs)
        edge = edges[favorite_idx]
        
        if edge > self.min_edge:
            # Standard Kelly formula
            kelly_bet = self.kelly_fraction * edge * bankroll
            return max(0, kelly_bet)
        
        return 0
    
    def _bet_contrarian(
        self, probs: List[float], odds: List[float], edges: List[float], 
        bankroll: float, confidence: float
    ) -> Tuple[int, float]:
        """
        Contrarian betting: bet on underdog or draw
        
        Mathematical approach:
        1. Identify favorite (highest probability)
        2. Calculate adjusted edges for non-favorites using bias multipliers
        3. Apply contrarian Kelly with reduced fraction
        4. Scale by inverse confidence (lower confidence = higher contrarian bet)
        """
        favorite_idx = np.argmax(probs)
        
        # Calculate adjusted edges for contrarian positions
        contrarian_edges = []
        for idx in range(3):
            if idx == favorite_idx:
                contrarian_edges.append(-999)  # Exclude favorite
            elif idx == 1:  # Draw
                # Apply draw bias multiplier
                adjusted_edge = edges[idx] * self.draw_bias_multiplier
                contrarian_edges.append(adjusted_edge)
            else:  # Underdog
                # Apply underdog bias multiplier
                adjusted_edge = edges[idx] * self.underdog_bias_multiplier
                contrarian_edges.append(adjusted_edge)
        
        # Find best contrarian opportunity
        best_contrarian_idx = np.argmax(contrarian_edges)
        best_contrarian_edge = contrarian_edges[best_contrarian_idx]
        
        # Only bet if adjusted edge is positive
        if best_contrarian_edge > self.min_edge:
            # Contrarian Kelly with confidence adjustment
            # Lower confidence = stronger contrarian signal
            confidence_factor = 1.0 - (confidence / self.low_confidence_threshold)
            confidence_factor = min(confidence_factor, 1.5)  # Cap at 1.5x
            
            kelly_bet = (
                self.contrarian_kelly_fraction 
                * best_contrarian_edge 
                * bankroll 
                * confidence_factor
            )
            
            return (best_contrarian_idx, max(0, kelly_bet))
        
        return None
    
    def _calculate_confidence(
        self, probs: List[float], home_id: str, away_id: str
    ) -> float:
        """
        Calculate prediction confidence
        
        Components:
        1. Probability separation (how clear is the favorite)
        2. Entropy (prediction uncertainty)
        3. Form consistency
        """
        # 1. Probability separation (0-1, higher = more confident)
        max_prob = max(probs)
        second_prob = sorted(probs, reverse=True)[1]
        prob_separation = max_prob - second_prob
        
        # 2. Entropy (0-1, lower entropy = more confident)
        # Normalize entropy to [0, 1]
        prob_entropy = entropy(probs) / np.log(3)  # Max entropy for 3 outcomes
        prediction_certainty = 1.0 - prob_entropy
        
        # 3. Form consistency (0-1, lower variance = more confident)
        form_consistency = 0.5  # Default
        if home_id in self.team_form and away_id in self.team_form:
            home_form_var = np.var(self.team_form[home_id]) if len(self.team_form[home_id]) > 1 else 0
            away_form_var = np.var(self.team_form[away_id]) if len(self.team_form[away_id]) > 1 else 0
            avg_variance = (home_form_var + away_form_var) / 2
            form_consistency = 1.0 / (1.0 + avg_variance)  # Lower variance = higher consistency
        
        # Weighted combination
        confidence = (
            0.4 * prob_separation +
            0.4 * prediction_certainty +
            0.2 * form_consistency
        )
        
        return confidence


class AdaptiveContrarianStrategy(EloModel):
    """
    Adaptive contrarian strategy with multiple threshold levels
    
    Uses progressive thresholds:
    - Very High Confidence (>0.75): Strong favorite bet
    - High Confidence (0.60-0.75): Moderate favorite bet
    - Medium Confidence (0.45-0.60): No bet
    - Low Confidence (0.30-0.45): Moderate contrarian bet
    - Very Low Confidence (<0.30): Strong contrarian bet
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        threshold_very_high: float = 0.75,
        threshold_high: float = 0.60,
        threshold_medium_low: float = 0.45,
        threshold_low: float = 0.30,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.20,
        draw_bias: float = 1.5,
        underdog_bias: float = 1.3,
        form_adjustment_weight: float = 0.1,
        streak_adjustment_weight: float = 0.05,
        initial_rating: float = 1500.0,
    ):
        super().__init__(
            k_factor=k_factor,
            home_advantage=home_advantage,
            rolling_window=rolling_window,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            form_adjustment_weight=form_adjustment_weight,
            streak_adjustment_weight=streak_adjustment_weight,
            initial_rating=initial_rating,
        )
        self.threshold_very_high = threshold_very_high
        self.threshold_high = threshold_high
        self.threshold_medium_low = threshold_medium_low
        self.threshold_low = threshold_low
        self.draw_bias = draw_bias
        self.underdog_bias = underdog_bias
    
    def _get_probabilities(self, home_id: int, away_id: int) -> List[float]:
        """
        Get win probabilities for home, away, and draw.
        Returns: [prob_home, prob_away, prob_draw]
        """
        home_rating = self._get_adjusted_rating(home_id, is_home=True)
        away_rating = self._get_adjusted_rating(away_id, is_home=False)
        prob_h, prob_a, prob_d = self._calculate_probabilities(home_rating, away_rating)
        return [prob_h, prob_a, prob_d]
        self.threshold_high = threshold_high
        self.threshold_medium_low = threshold_medium_low
        self.threshold_low = threshold_low
        self.draw_bias = draw_bias
        self.underdog_bias = underdog_bias
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets using adaptive thresholds.
        Overrides EloModel.place_bets to implement adaptive contrarian logic.
        """
        # Update ratings with new game results
        if not inc.empty:
            self._update_ratings(inc)
        
        # Initialize empty bets
        bets = pd.DataFrame(
            0.0,
            index=opps.index,
            columns=["BetH", "BetA", "BetD"]
        )
        
        if opps.empty:
            return bets
        
        # Get betting constraints
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        
        # Calculate bets for each opportunity
        for idx, row in opps.iterrows():
            home_id = row["HID"]
            away_id = row["AID"]
            odds_h = row["OddsH"]
            odds_a = row["OddsA"]
            odds_d = row["OddsD"]
            
            # Skip if no valid odds
            if odds_h == 0 or odds_a == 0 or odds_d == 0:
                continue
            
            # Ensure teams have ratings
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            probs = self._get_probabilities(home_id, away_id)
            edges = [
                probs[0] * odds_h - 1,
                probs[2] * odds_d - 1,
                probs[1] * odds_a - 1
            ]
            
            confidence = self._calculate_confidence(probs, home_id, away_id)
            
            # Progressive betting based on confidence levels
            if confidence >= self.threshold_very_high:
                # Very high confidence: Strong favorite bet (30% Kelly)
                favorite_idx = np.argmax(probs)
                if edges[favorite_idx] > self.min_edge:
                    bet = 0.30 * edges[favorite_idx] * bankroll
                    if bet > min_bet:
                        if favorite_idx == 0:
                            bets.at[idx, "BetH"] = bet
                        elif favorite_idx == 1:
                            bets.at[idx, "BetA"] = bet
                        else:
                            bets.at[idx, "BetD"] = bet
            
            elif confidence >= self.threshold_high:
                # High confidence: Moderate favorite bet (20% Kelly)
                favorite_idx = np.argmax(probs)
                if edges[favorite_idx] > self.min_edge:
                    bet = 0.20 * edges[favorite_idx] * bankroll
                    if bet > min_bet:
                        if favorite_idx == 0:
                            bets.at[idx, "BetH"] = bet
                        elif favorite_idx == 1:
                            bets.at[idx, "BetA"] = bet
                        else:
                            bets.at[idx, "BetD"] = bet
            
            elif confidence >= self.threshold_medium_low:
                # Medium confidence: No bet (too uncertain)
                pass
            
            elif confidence >= self.threshold_low:
                # Low confidence: Moderate contrarian (10% Kelly)
                result = self._place_contrarian_bet(
                    probs, [odds_h, odds_a, odds_d], edges, bankroll, 0.10, confidence
                )
                if result:
                    bet_idx, bet = result
                    if bet > min_bet:
                        if bet_idx == 0:
                            bets.at[idx, "BetH"] = bet
                        elif bet_idx == 1:
                            bets.at[idx, "BetD"] = bet
                        else:
                            bets.at[idx, "BetA"] = bet
            
            else:
                # Very low confidence: Strong contrarian (15% Kelly)
                result = self._place_contrarian_bet(
                    probs, [odds_h, odds_a, odds_d], edges, bankroll, 0.15, confidence
                )
                if result:
                    bet_idx, bet = result
                    if bet > min_bet:
                        if bet_idx == 0:
                            bets.at[idx, "BetH"] = bet
                        elif bet_idx == 1:
                            bets.at[idx, "BetD"] = bet
                        else:
                            bets.at[idx, "BetA"] = bet
        
        # Scale if needed
        total_bet = bets[["BetH", "BetA", "BetD"]].sum().sum()
        if total_bet > bankroll:
            scale_factor = (bankroll * 0.95) / total_bet
            bets[["BetH", "BetA", "BetD"]] *= scale_factor
            # Re-apply min_bet filter
            bets.loc[bets["BetH"] < min_bet, "BetH"] = 0
            bets.loc[bets["BetA"] < min_bet, "BetA"] = 0
            bets.loc[bets["BetD"] < min_bet, "BetD"] = 0
        
        return bets
    
    def _place_contrarian_bet(
        self, probs: List[float], odds: List[float], edges: List[float],
        bankroll: float, kelly_frac: float, confidence: float
    ) -> Tuple[int, float]:
        """Place contrarian bet on underdog or draw"""
        favorite_idx = np.argmax(probs)
        
        # Evaluate contrarian options
        contrarian_values = []
        for idx in range(3):
            if idx == favorite_idx:
                contrarian_values.append(-999)
            elif idx == 1:  # Draw
                contrarian_values.append(edges[idx] * self.draw_bias)
            else:  # Underdog
                contrarian_values.append(edges[idx] * self.underdog_bias)
        
        best_idx = np.argmax(contrarian_values)
        best_edge = contrarian_values[best_idx]
        
        if best_edge > self.min_edge:
            # Inverse confidence factor for contrarian bets
            conf_factor = 1.0 + (1.0 - confidence)
            bet = kelly_frac * best_edge * bankroll * conf_factor
            return (best_idx, max(0, bet))
        
        return None
    
    def _calculate_confidence(
        self, probs: List[float], home_id: str, away_id: str
    ) -> float:
        """Calculate prediction confidence"""
        # Probability separation
        max_prob = max(probs)
        second_prob = sorted(probs, reverse=True)[1]
        prob_separation = max_prob - second_prob
        
        # Entropy-based certainty
        prob_entropy = entropy(probs) / np.log(3)
        prediction_certainty = 1.0 - prob_entropy
        
        # Form consistency
        form_consistency = 0.5
        if home_id in self.team_form and away_id in self.team_form:
            home_var = np.var(self.team_form[home_id]) if len(self.team_form[home_id]) > 1 else 0
            away_var = np.var(self.team_form[away_id]) if len(self.team_form[away_id]) > 1 else 0
            avg_var = (home_var + away_var) / 2
            form_consistency = 1.0 / (1.0 + avg_var)
        
        confidence = (
            0.4 * prob_separation +
            0.4 * prediction_certainty +
            0.2 * form_consistency
        )
        
        return confidence


def get_contrarian_strategy(strategy_name: str, games_data: pd.DataFrame, **kwargs) -> EloModel:
    """
    Factory function to create contrarian strategies
    
    Args:
        strategy_name: Name of the strategy
        games_data: Historical games data for initialization
        **kwargs: Additional parameters for the strategy
    
    Returns:
        Initialized strategy instance
    """
    strategies = {
        "contrarian": ContrarianBettingStrategy,
        "adaptive_contrarian": AdaptiveContrarianStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    return strategy_class(**kwargs)
