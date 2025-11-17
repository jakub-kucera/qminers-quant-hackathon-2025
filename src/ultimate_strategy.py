"""
Ultimate Betting Strategy - Combines all learnings for maximum bankroll

This strategy integrates:
1. Value Betting selectivity (only positive strategy from research)
2. Ultra-Conservative Contrarian thresholds (best contrarian performer)
3. Optimized Kelly Criterion sizing (best capital preservation)
4. Confidence-based bet scaling
5. Adaptive risk management with stop-loss protection
"""

import pandas as pd
import numpy as np
from models import EloModel


class UltimateStrategy(EloModel):
    """
    Ultimate betting strategy combining all best practices:
    - Value betting on high confidence with proven edge
    - Contrarian betting on ultra-low confidence (exploit favorite-longshot bias)
    - Highly conservative Kelly fractions for capital preservation
    - Multi-stage filtering to only take highest quality bets
    - Dynamic position sizing based on confidence and edge quality
    """
    
    def __init__(
        self,
        # Elo parameters (optimized from Kelly research)
        k_factor=21.85,
        home_advantage=190.14,
        rolling_window=10,
        form_adjustment_weight=0.15,
        streak_adjustment_weight=0.10,
        initial_rating=1500,
        
        # Value betting thresholds (balanced for finding opportunities)
        high_confidence_threshold=0.65,  # High bar but achievable
        value_min_edge=0.08,  # 8% minimum edge
        value_kelly_fraction=0.12,  # Moderate Kelly for growth
        
        # Contrarian thresholds (from Ultra-Conservative but relaxed)
        ultra_low_confidence_threshold=0.35,  # Capture uncertain games
        contrarian_min_edge=0.06,  # Still require positive edge
        contrarian_kelly_fraction=0.08,  # Conservative but growth-oriented
        
        # Bias multipliers (proven from research)
        draw_bias=1.5,  # Moderate draw bias
        underdog_bias=1.3,  # Moderate underdog bias
        
        # Risk management (from best practices)
        max_bet_fraction=0.15,  # Never risk more than 15% on one bet
        min_bet=10.0,  # Minimum bet size
        stop_loss_threshold=0.85,  # Stop if bankroll drops to 85%
    ):
        super().__init__(
            k_factor=k_factor,
            home_advantage=home_advantage,
            rolling_window=int(rolling_window),
            min_edge=0.05,  # Low minimum for initial filtering
            kelly_fraction=0.10,  # Default, will be overridden
            form_adjustment_weight=form_adjustment_weight,
            streak_adjustment_weight=streak_adjustment_weight,
            initial_rating=initial_rating,
        )
        
        # Value betting parameters (relaxed from initial conservative settings)
        self.high_confidence_threshold = 0.65  # Lowered from 0.75 to find more opportunities
        self.value_min_edge = 0.08  # Lowered from 0.12 to 8%
        self.value_kelly_fraction = 0.12  # Increased from 0.08 for higher returns
        
        # Contrarian parameters (from Ultra-Conservative but slightly relaxed)
        self.ultra_low_confidence_threshold = 0.35  # Raised from 0.30 to capture more
        self.contrarian_min_edge = 0.06  # Lowered from 0.08
        self.contrarian_kelly_fraction = 0.08  # Increased from 0.05
        
        # Bias multipliers (proven effective from research)
        self.draw_bias = 1.5  # Increased from 1.3
        self.underdog_bias = 1.3  # Increased from 1.2
        
        # Risk management
        self.max_bet_fraction = max_bet_fraction
        self.min_bet = min_bet
        self.stop_loss_threshold = stop_loss_threshold
        self.initial_bankroll = None
        
    def _calculate_confidence(self, probs):
        """Calculate prediction confidence"""
        # Probability separation (how clear the favorite is)
        max_prob = max(probs)
        second_max = sorted(probs, reverse=True)[1]
        prob_separation = max_prob - second_max
        
        # Prediction certainty (inverse of entropy)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = -np.log(1/3)  # Maximum for 3 outcomes
        certainty = 1 - (entropy / max_entropy)
        
        # Combined confidence (weighted average)
        confidence = 0.6 * prob_separation + 0.4 * certainty
        return confidence
    
    def place_bets(self, summary, upcoming_games, inc):
        """
        Place bets using ultimate strategy:
        1. Value betting on high confidence
        2. Contrarian betting on ultra-low confidence  
        3. No betting in medium confidence zone
        """
        # Update ratings with new game results
        if not inc.empty:
            self._update_ratings(inc)
            
        bankroll = summary["Bankroll"].iloc[0]
        
        if self.initial_bankroll is None:
            self.initial_bankroll = bankroll
            
        # Check stop-loss
        if bankroll < self.initial_bankroll * self.stop_loss_threshold:
            # Emergency stop - no more bets
            return pd.DataFrame(0.0, index=upcoming_games.index, columns=["BetH", "BetA", "BetD"])
        
        # Initialize bets DataFrame
        bets = pd.DataFrame(0.0, index=upcoming_games.index, columns=["BetH", "BetA", "BetD"])
        
        for idx, game in upcoming_games.iterrows():
            home_id = game["HID"]
            away_id = game["AID"]
            
            # Get probabilities
            probs = self._get_probabilities(game)
            if probs is None:
                continue
                
            prob_home, prob_draw, prob_away = probs
            
            # Get odds
            odds_home = game.get("OddsH", 0)
            odds_draw = game.get("OddsD", 0) 
            odds_away = game.get("OddsA", 0)
            
            if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
                continue
            
            # Calculate confidence
            confidence = self._calculate_confidence([prob_home, prob_draw, prob_away])
            
            # Calculate edges
            edge_home = prob_home * odds_home - 1
            edge_draw = prob_draw * odds_draw - 1
            edge_away = prob_away * odds_away - 1
            
            # STRATEGY 1: High-confidence value betting on favorite
            if confidence >= self.high_confidence_threshold:
                # Identify favorite and bet on it if edge is sufficient
                if prob_home >= prob_draw and prob_home >= prob_away and edge_home >= self.value_min_edge:
                    # Home is favorite with good edge
                    kelly_bet = self.value_kelly_fraction * edge_home * bankroll
                    confidence_multiplier = 1.0 + (confidence - self.high_confidence_threshold) * 2.0
                    bet_size = kelly_bet * confidence_multiplier
                    bets.loc[idx, "BetH"] = min(bet_size, bankroll * self.max_bet_fraction)
                        
                elif prob_away >= prob_home and prob_away >= prob_draw and edge_away >= self.value_min_edge:
                    # Away is favorite with good edge
                    kelly_bet = self.value_kelly_fraction * edge_away * bankroll
                    confidence_multiplier = 1.0 + (confidence - self.high_confidence_threshold) * 2.0
                    bet_size = kelly_bet * confidence_multiplier
                    bets.loc[idx, "BetA"] = min(bet_size, bankroll * self.max_bet_fraction)
            
            # STRATEGY 2: Ultra-low confidence contrarian betting
            elif confidence <= self.ultra_low_confidence_threshold:
                # Bet against the "favorite" or on draw with bias multipliers
                adjusted_edge_draw = edge_draw * self.draw_bias
                adjusted_edge_home = edge_home * self.underdog_bias if prob_home < prob_away else edge_home
                adjusted_edge_away = edge_away * self.underdog_bias if prob_away < prob_home else edge_away
                
                # Find best contrarian opportunity
                best_edge = max(adjusted_edge_draw, adjusted_edge_home, adjusted_edge_away)
                
                if best_edge >= self.contrarian_min_edge:
                    # Inverse confidence scaling (lower confidence = larger bet)
                    inverse_confidence = (self.ultra_low_confidence_threshold - confidence) / self.ultra_low_confidence_threshold
                    kelly_bet = self.contrarian_kelly_fraction * best_edge * bankroll
                    bet_size = kelly_bet * (1.0 + inverse_confidence)
                    bet_size = min(bet_size, bankroll * self.max_bet_fraction)
                    
                    # Place bet on best opportunity
                    if adjusted_edge_draw == best_edge:
                        bets.loc[idx, "BetD"] = bet_size
                    elif adjusted_edge_home == best_edge:
                        bets.loc[idx, "BetH"] = bet_size
                    else:
                        bets.loc[idx, "BetA"] = bet_size
            
            # STRATEGY 3: Medium confidence - NO BET (too uncertain)
            # This is intentional - avoid the dangerous middle ground
        
        # Apply minimum bet filter
        bets[bets < self.min_bet] = 0
        
        # Final safety check: don't exceed bankroll
        total_bets = bets.sum().sum()
        if total_bets > bankroll * 0.95:
            bets *= (bankroll * 0.95) / total_bets
            bets[bets < self.min_bet] = 0  # Re-apply min bet after scaling
        
        return bets
    
    def _get_probabilities(self, game):
        """Get win/draw/loss probabilities for a game"""
        home_id = game["HID"]
        away_id = game["AID"]
        
        if home_id not in self.ratings or away_id not in self.ratings:
            return None
        
        # Get base ratings
        home_rating = self.ratings[home_id]
        away_rating = self.ratings[away_id]
        
        # Apply home advantage
        home_rating += self.home_advantage
        
        # Apply form adjustments
        if home_id in self.team_form and len(self.team_form[home_id]) > 0:
            home_form = sum(self.team_form[home_id]) / len(self.team_form[home_id])
            home_rating += home_form * self.form_adjustment_weight * 400
        
        if away_id in self.team_form and len(self.team_form[away_id]) > 0:
            away_form = sum(self.team_form[away_id]) / len(self.team_form[away_id])
            away_rating += away_form * self.form_adjustment_weight * 400
        
        # Apply streak adjustments  
        if home_id in self.team_streaks:
            home_rating += self.team_streaks[home_id] * self.streak_adjustment_weight * 50
        if away_id in self.team_streaks:
            away_rating += self.team_streaks[away_id] * self.streak_adjustment_weight * 50
        
        # Calculate probabilities using Elo formula
        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        expected_away = 1 - expected_home
        
        # Estimate draw probability (conservative approach)
        draw_factor = 0.22  # Slightly lower than default for more decisive predictions
        prob_draw = draw_factor * min(expected_home, expected_away)
        prob_home = expected_home * (1 - prob_draw)
        prob_away = expected_away * (1 - prob_draw)
        
        # Normalize
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        return (prob_home, prob_draw, prob_away)


def get_ultimate_strategy(games_data):
    """Factory function to create ultimate strategy with optimal parameters"""
    return UltimateStrategy(
        # Optimized Elo parameters (from Kelly optimization)
        k_factor=21.85,
        home_advantage=190.14,
        rolling_window=10,
        form_adjustment_weight=0.15,
        streak_adjustment_weight=0.10,
        initial_rating=1500,
        
        # Value betting (from Value Betting research - only positive strategy)
        high_confidence_threshold=0.75,
        value_min_edge=0.12,
        value_kelly_fraction=0.08,
        
        # Contrarian (from Ultra-Conservative Contrarian - best performer)
        ultra_low_confidence_threshold=0.30,
        contrarian_min_edge=0.08,
        contrarian_kelly_fraction=0.05,
        
        # Conservative biases
        draw_bias=1.3,
        underdog_bias=1.2,
        
        # Risk management
        max_bet_fraction=0.15,
        min_bet=10.0,
        stop_loss_threshold=0.85,
    )
