"""
Hybrid Betting Strategies - Combining best performing strategies
Leverages Value Betting, Optimized Fractional Kelly, and adaptive switching
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from models import EloModel
from betting_strategies_research import (
    ValueBettingStrategy,
    FullKellyStrategy,
)


class HybridValueKellyStrategy(EloModel):
    """
    Hybrid strategy combining Value Betting's selectivity with Kelly's bet sizing
    
    Approach:
    - Uses Value Betting logic to identify high-EV opportunities
    - Uses Fractional Kelly for optimal bet sizing
    - Adds confidence thresholds for additional safety
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        min_edge: float = 0.08,  # Higher threshold for selectivity
        kelly_fraction: float = 0.15,  # Conservative Kelly
        value_multiplier: float = 2.0,  # Bet size scales with value
        confidence_threshold: float = 0.60,  # Only bet on confident predictions
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
        self.value_multiplier = value_multiplier
        self.confidence_threshold = confidence_threshold
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets using hybrid Value-Kelly approach
        
        Args:
            summary: Summary DataFrame with bankroll and betting constraints
            opps: Opportunities DataFrame with available betting options
            inc: Incremental games DataFrame with recent game results
            
        Returns:
            DataFrame with bet amounts for each opportunity
        """
        if opps.empty:
            return pd.DataFrame()
        
        # Get bankroll from summary
        bankroll = summary["Bankroll"].iloc[0] if not summary.empty else 1000.0
        min_bet = summary.get("MinBet", pd.Series([10])).iloc[0] if not summary.empty else 10.0
        
        n_games = len(opps)
        bets = np.zeros((n_games, 3))  # [home, draw, away]
        
        for i, (_, game) in enumerate(opps.iterrows()):
            home_id = game["HID"]
            away_id = game["AID"]
            
            # Get probabilities
            probs = self._calculate_probabilities(home_id, away_id)
            
            # Get odds
            odds = [game.get("OddsH", 0), game.get("OddsD", 0), game.get("OddsA", 0)]
            if any(o == 0 for o in odds):
                continue
            
            # Calculate edges
            edges = [probs[j] * odds[j] - 1 for j in range(3)]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(probs, home_id, away_id)
            
            # Only bet if confident and positive edge
            if confidence < self.confidence_threshold:
                continue
            
            # Find best opportunity
            best_idx = np.argmax(edges)
            best_edge = edges[best_idx]
            
            if best_edge > self.min_edge:
                # Value-based selection: higher edges get priority
                value_score = best_edge / self.min_edge  # Relative value
                
                # Kelly bet sizing with value multiplier
                kelly_bet = (
                    self.kelly_fraction
                    * best_edge
                    * bankroll
                    * min(self.value_multiplier, value_score)  # Scale by value
                    * confidence  # Scale by confidence
                )
                
                # Apply constraints
                kelly_bet = max(min_bet, min(kelly_bet, bankroll * 0.1))  # Max 10% per bet
                
                bets[i, best_idx] = kelly_bet
        
        # Ensure total bets don't exceed bankroll
        total_bet = bets.sum()
        if total_bet > bankroll:
            bets *= (bankroll * 0.95) / total_bet
            bets[bets < min_bet] = 0
        
        # Convert to DataFrame matching EloModel format
        bet_df = pd.DataFrame(
            bets,
            index=opps.index,
            columns=["H", "D", "A"]
        )
        return bet_df
    
    def _calculate_confidence(
        self, probs: List[float], home_id: str, away_id: str
    ) -> float:
        """Calculate confidence score for the prediction"""
        # Probability separation (how clear is the winner)
        sorted_probs = sorted(probs, reverse=True)
        separation = sorted_probs[0] - sorted_probs[1]
        
        # Entropy (lower = more certain)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log(3)
        certainty = 1 - (entropy / max_entropy)
        
        # Form consistency
        home_form_var = np.var(self.team_form.get(home_id, [0]))
        away_form_var = np.var(self.team_form.get(away_id, [0]))
        form_consistency = 1 / (1 + home_form_var + away_form_var)
        
        # Combine factors
        confidence = (
            0.4 * separation +
            0.4 * certainty +
            0.2 * form_consistency
        )
        
        return confidence


class AdaptiveHybridStrategy(EloModel):
    """
    Adaptive strategy that switches between Value Betting and Conservative Kelly
    based on performance and market conditions
    
    Switches strategy based on:
    - Current bankroll vs initial (performance)
    - Recent win rate
    - Market conditions (volatility)
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        aggressive_min_edge: float = 0.05,  # For value betting mode
        conservative_min_edge: float = 0.10,  # For conservative mode
        aggressive_kelly: float = 0.25,
        conservative_kelly: float = 0.10,
        switch_threshold: float = 0.90,  # Switch to conservative below 90% of initial
        form_adjustment_weight: float = 0.1,
        streak_adjustment_weight: float = 0.05,
        initial_rating: float = 1500.0,
    ):
        super().__init__(
            k_factor=k_factor,
            home_advantage=home_advantage,
            rolling_window=rolling_window,
            min_edge=aggressive_min_edge,
            kelly_fraction=aggressive_kelly,
            form_adjustment_weight=form_adjustment_weight,
            streak_adjustment_weight=streak_adjustment_weight,
            initial_rating=initial_rating,
        )
        self.aggressive_min_edge = aggressive_min_edge
        self.conservative_min_edge = conservative_min_edge
        self.aggressive_kelly = aggressive_kelly
        self.conservative_kelly = conservative_kelly
        self.switch_threshold = switch_threshold
        self.initial_bankroll = None
        self.recent_results = []  # Track recent wins/losses
        self.mode = "aggressive"  # Start aggressive
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets with adaptive strategy switching
        
        Args:
            summary: Summary DataFrame with bankroll and betting constraints
            opps: Opportunities DataFrame with available betting options
            inc: Incremental games DataFrame with recent game results
            
        Returns:
            DataFrame with bet amounts for each opportunity
        """
        if opps.empty:
            return pd.DataFrame()
        
        # Get bankroll from summary
        bankroll = summary["Bankroll"].iloc[0] if not summary.empty else 1000.0
        min_bet = summary.get("MinBet", pd.Series([10])).iloc[0] if not summary.empty else 10.0
        
        # Initialize tracking
        if self.initial_bankroll is None:
            self.initial_bankroll = bankroll
        
        # Determine current mode
        performance = bankroll / self.initial_bankroll
        recent_win_rate = self._get_recent_win_rate()
        
        if performance < self.switch_threshold or recent_win_rate < 0.45:
            # Switch to conservative mode
            self.mode = "conservative"
            self.min_edge = self.conservative_min_edge
            self.kelly_fraction = self.conservative_kelly
        else:
            # Use aggressive mode
            self.mode = "aggressive"
            self.min_edge = self.aggressive_min_edge
            self.kelly_fraction = self.aggressive_kelly
        
        # Place bets using parent logic with adjusted parameters
        bets = super().place_bets(summary, opps, inc)
        
        return bets
    
    def update(self, game: Dict[str, Any]):
        """Update model and track results"""
        # Track result before updating
        home_id = game["HID"]
        away_id = game["AID"]
        
        home_score = game.get("HS", 0)
        away_score = game.get("AS", 0)
        
        # Determine outcome
        if home_score > away_score:
            result = "home"
        elif away_score > home_score:
            result = "away"
        else:
            result = "draw"
        
        # Get our prediction
        probs = self._calculate_probabilities(home_id, away_id)
        predicted = ["home", "draw", "away"][np.argmax(probs)]
        
        # Track if we were correct
        self.recent_results.append(1 if result == predicted else 0)
        if len(self.recent_results) > 50:  # Keep last 50 results
            self.recent_results.pop(0)
        
        # Update model
        super().update(game)
    
    def _get_recent_win_rate(self) -> float:
        """Calculate recent prediction win rate"""
        if len(self.recent_results) < 10:
            return 0.5  # Default to neutral
        return np.mean(self.recent_results[-30:])  # Last 30 games


class EnsembleStrategy(EloModel):
    """
    Ensemble of multiple strategies with weighted voting
    
    Combines:
    - Value Betting (for selectivity)
    - Fractional Kelly (for bet sizing)
    - Hybrid Value-Kelly (for balanced approach)
    
    Uses weighted average based on strategy historical performance
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        min_edge: float = 0.06,
        kelly_fraction: float = 0.15,
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
        
        # Initialize sub-strategies
        self.strategies = {
            "value": ValueBettingStrategy(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_ev=0.10,
                max_bet_percent=0.05,
            ),
            "kelly": FullKellyStrategy(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_edge=0.10,
            ),
            "hybrid": HybridValueKellyStrategy(
                k_factor=k_factor,
                home_advantage=home_advantage,
                rolling_window=rolling_window,
                min_edge=0.06,
                kelly_fraction=0.15,
            ),
        }
        
        # Track strategy performance
        self.strategy_weights = {
            "value": 0.40,  # Higher weight to best performer
            "kelly": 0.30,
            "hybrid": 0.30,
        }
        
        self.strategy_performance = {k: [] for k in self.strategies.keys()}
    
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets using ensemble of strategies
        
        Args:
            summary: Summary DataFrame with bankroll and betting constraints
            opps: Opportunities DataFrame with available betting options
            inc: Incremental games DataFrame with recent game results
            
        Returns:
            DataFrame with bet amounts for each opportunity
        """
        if opps.empty:
            return pd.DataFrame()
        
        # Get bets from each strategy
        strategy_bets = {}
        for name, strategy in self.strategies.items():
            # Sync ratings with main model
            strategy.ratings = self.ratings.copy()
            strategy.team_form = {k: list(v) for k, v in self.team_form.items()}
            strategy.team_streaks = self.team_streaks.copy()
            
            strategy_bets[name] = strategy.place_bets(summary, opps, inc)
        
        # Weighted combination
        n_games = len(opps)
        combined_bets = np.zeros((n_games, 3))
        
        # Get bankroll and min_bet from summary
        bankroll = summary["Bankroll"].iloc[0] if not summary.empty else 1000.0
        min_bet = summary.get("MinBet", pd.Series([10])).iloc[0] if not summary.empty else 10.0
        
        for name, bets in strategy_bets.items():
            # Convert DataFrame to numpy if needed
            if isinstance(bets, pd.DataFrame):
                bets_array = bets[["H", "D", "A"]].values
            else:
                bets_array = bets
            combined_bets += bets_array * self.strategy_weights[name]
        
        # Apply consensus filter: only bet if multiple strategies agree
        for i in range(n_games):
            # Count how many strategies want to bet on this game
            strategies_betting = sum(
                1 for bets in strategy_bets.values() 
                if (bets.iloc[i].sum() if isinstance(bets, pd.DataFrame) else bets[i].sum()) > 0
            )
            
            # Require at least 2 strategies to agree
            if strategies_betting < 2:
                combined_bets[i] = 0
        
        # Ensure valid bets
        combined_bets[combined_bets < min_bet] = 0
        
        # Ensure total doesn't exceed bankroll
        total_bet = combined_bets.sum()
        if total_bet > bankroll:
            combined_bets *= (bankroll * 0.95) / total_bet
            combined_bets[combined_bets < min_bet] = 0
        
        # Convert to DataFrame matching EloModel format
        bet_df = pd.DataFrame(
            combined_bets,
            index=opps.index,
            columns=["H", "D", "A"]
        )
        return bet_df
    
    def update(self, game: Dict[str, Any]):
        """Update all strategies"""
        super().update(game)
        for strategy in self.strategies.values():
            strategy.update(game)


def get_hybrid_strategy(
    strategy_name: str,
    games_data: pd.DataFrame,
    **kwargs
) -> EloModel:
    """
    Factory function to get hybrid betting strategies
    
    Args:
        strategy_name: One of ['hybrid_value_kelly', 'adaptive_hybrid', 'ensemble']
        games_data: Historical games data for initialization
        **kwargs: Additional parameters for the strategy
    
    Returns:
        Initialized hybrid strategy model
    """
    
    strategies = {
        "hybrid_value_kelly": HybridValueKellyStrategy,
        "adaptive_hybrid": AdaptiveHybridStrategy,
        "ensemble": EnsembleStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(strategies.keys())}"
        )
    
    # Create and initialize strategy
    strategy = strategies[strategy_name](**kwargs)
    
    # Initialize with historical data
    for _, game in games_data.iterrows():
        if "HS" in game and "AS" in game:
            strategy.update(game)
    
    return strategy
