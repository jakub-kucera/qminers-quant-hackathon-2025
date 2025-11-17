"""
Model implementations for betting strategy optimization.

This module provides:
- BaseModel: Abstract base class with shared functionality
- EloModel: Elo rating-based betting model with home advantage and form tracking
- AdvancedEloModel: Enhanced Elo model with additional features
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for betting models.
    
    Provides shared functionality for all models including:
    - Parameter management
    - Bet validation utilities
    - Common betting logic
    """
    
    def __init__(self, **params):
        """
        Initialize model with parameters.
        
        Args:
            **params: Model-specific hyperparameters
        """
        self.params = params
        self.history = []
        
    @abstractmethod
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Place bets based on current opportunities and game history.
        
        Args:
            summary: Summary DataFrame with bankroll and betting constraints
            opps: Opportunities DataFrame with available betting options
            inc: Incremental games DataFrame with recent game results
            
        Returns:
            DataFrame with bet amounts for each opportunity
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        self.params.update(params)


class EloModel(BaseModel):
    """
    Elo rating-based betting model.
    
    Features:
    - Elo ratings for each team
    - Home advantage factor
    - Rolling average performance
    - Win/loss streak tracking
    - Kelly Criterion for bet sizing
    """
    
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        rolling_window: int = 10,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        form_adjustment_weight: float = 0.1,
        streak_adjustment_weight: float = 0.05,
        initial_rating: float = 1500.0,
    ):
        """
        Initialize Elo model with hyperparameters.
        
        Args:
            k_factor: Elo K-factor for rating updates
            home_advantage: Home team rating advantage
            rolling_window: Window size for rolling averages
            min_edge: Minimum edge required to place bet
            kelly_fraction: Fraction of Kelly Criterion to use
            form_adjustment_weight: Weight for recent form adjustment
            streak_adjustment_weight: Weight for streak adjustment
            initial_rating: Initial Elo rating for all teams
        """
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
        
        # Team ratings
        self.ratings = {}
        
        # Team statistics
        self.team_form = {}  # Rolling average of recent performance
        self.team_streaks = {}  # Current win/loss streaks
        
        # Initialize from params
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.rolling_window = rolling_window
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.form_adjustment_weight = form_adjustment_weight
        self.streak_adjustment_weight = streak_adjustment_weight
        self.initial_rating = initial_rating
        
    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Place bets using Elo ratings and Kelly Criterion."""
        
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
            
            # Ensure teams have ratings
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
            
            # Calculate win probabilities
            prob_h, prob_a, prob_d = self._calculate_probabilities(
                home_rating, away_rating
            )
            
            # Calculate edge for each outcome
            edge_h = prob_h - (1.0 / odds_h) if odds_h > 0 else -1
            edge_a = prob_a - (1.0 / odds_a) if odds_a > 0 else -1
            edge_d = prob_d - (1.0 / odds_d) if odds_d > 0 else -1
            
            # Find best bet
            best_edge = max(edge_h, edge_a, edge_d)
            
            # Only bet if edge exceeds minimum
            if best_edge > self.min_edge:
                # Determine which outcome to bet on
                if best_edge == edge_h and odds_h > 0:
                    bet_amount = self._kelly_bet(prob_h, odds_h, bankroll)
                    bets.loc[idx, "BetH"] = bet_amount
                elif best_edge == edge_a and odds_a > 0:
                    bet_amount = self._kelly_bet(prob_a, odds_a, bankroll)
                    bets.loc[idx, "BetA"] = bet_amount
                elif best_edge == edge_d and odds_d > 0:
                    bet_amount = self._kelly_bet(prob_d, odds_d, bankroll)
                    bets.loc[idx, "BetD"] = bet_amount
        
        # Ensure bets are within constraints
        bets = bets.clip(lower=0)
        bets[bets < min_bet] = 0
        bets = bets.clip(upper=max_bet)
        
        # Don't bet more than available bankroll
        total_bet = bets.sum().sum()
        if total_bet > bankroll:
            bets *= (bankroll * 0.95) / total_bet  # Use 95% to be safe
        
        return bets
    
    def _update_ratings(self, games: pd.DataFrame) -> None:
        """Update Elo ratings based on game results."""
        for idx, game in games.iterrows():
            home_id = game["HID"]
            away_id = game["AID"]
            home_won = game["H"]
            away_won = game["A"]
            draw = game["D"]
            
            # Initialize if needed
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            # Get current ratings
            home_rating = self.ratings[home_id] + self.home_advantage
            away_rating = self.ratings[away_id]
            
            # Expected scores
            expected_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400))
            expected_away = 1.0 - expected_home
            
            # Actual scores
            if home_won:
                actual_home, actual_away = 1.0, 0.0
            elif away_won:
                actual_home, actual_away = 0.0, 1.0
            else:  # draw
                actual_home, actual_away = 0.5, 0.5
            
            # Update ratings
            self.ratings[home_id] += self.k_factor * (actual_home - expected_home)
            self.ratings[away_id] += self.k_factor * (actual_away - expected_away)
            
            # Update form (rolling average of results)
            self.team_form[home_id].append(actual_home)
            if len(self.team_form[home_id]) > self.rolling_window:
                self.team_form[home_id].pop(0)
            
            self.team_form[away_id].append(actual_away)
            if len(self.team_form[away_id]) > self.rolling_window:
                self.team_form[away_id].pop(0)
            
            # Update streaks
            if home_won:
                self.team_streaks[home_id] = max(1, self.team_streaks[home_id] + 1)
                self.team_streaks[away_id] = min(-1, self.team_streaks[away_id] - 1)
            elif away_won:
                self.team_streaks[home_id] = min(-1, self.team_streaks[home_id] - 1)
                self.team_streaks[away_id] = max(1, self.team_streaks[away_id] + 1)
            else:  # draw
                self.team_streaks[home_id] = 0
                self.team_streaks[away_id] = 0
    
    def _get_adjusted_rating(self, team_id: int, is_home: bool) -> float:
        """Get rating adjusted for form, streak, and home advantage."""
        base_rating = self.ratings.get(team_id, self.initial_rating)
        
        # Home advantage
        if is_home:
            base_rating += self.home_advantage
        
        # Form adjustment
        form = self.team_form.get(team_id, [])
        if form:
            avg_form = np.mean(form)
            # Scale form from [0,1] to [-200, 200]
            form_adjustment = (avg_form - 0.5) * 400 * self.form_adjustment_weight
            base_rating += form_adjustment
        
        # Streak adjustment
        streak = self.team_streaks.get(team_id, 0)
        streak_adjustment = streak * 20 * self.streak_adjustment_weight
        base_rating += streak_adjustment
        
        return base_rating
    
    def _calculate_probabilities(
        self, home_rating: float, away_rating: float
    ) -> tuple:
        """
        Calculate win probabilities for home, away, and draw.
        
        Uses Elo formula with draw probability estimation.
        """
        # Expected win probability for home (without considering draws)
        rating_diff = home_rating - away_rating
        prob_home_vs_away = 1.0 / (1.0 + 10 ** (-rating_diff / 400))
        
        # Estimate draw probability based on closeness of ratings
        # Draws more likely when teams are evenly matched
        draw_factor = 0.25  # Base draw probability
        rating_closeness = 1.0 - min(abs(rating_diff) / 400, 1.0)
        prob_d = draw_factor * rating_closeness
        
        # Adjust win probabilities accounting for draws
        prob_h = prob_home_vs_away * (1.0 - prob_d)
        prob_a = (1.0 - prob_home_vs_away) * (1.0 - prob_d)
        
        # Normalize to ensure they sum to 1
        total = prob_h + prob_a + prob_d
        prob_h /= total
        prob_a /= total
        prob_d /= total
        
        return prob_h, prob_a, prob_d
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """
        Calculate bet size using Kelly Criterion.
        
        Args:
            prob: Estimated probability of winning
            odds: Decimal odds
            bankroll: Available bankroll
            
        Returns:
            Bet amount (fractional Kelly)
        """
        if odds <= 1.0 or prob <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = prob, q = 1 - prob
        b = odds - 1
        kelly = (b * prob - (1 - prob)) / b
        
        # Only bet if kelly is positive
        if kelly <= 0:
            return 0.0
        
        # Use fractional Kelly for safety
        bet_fraction = kelly * self.kelly_fraction
        bet_amount = bankroll * bet_fraction
        
        return max(0, bet_amount)


class AdvancedEloModel(EloModel):
    """
    Advanced Elo model with additional features.
    
    Could be extended with:
    - Score differential consideration
    - Recent performance metrics (shots, saves, etc.)
    - Season-specific adjustments
    - More sophisticated draw probability estimation
    """
    
    def __init__(
        self,
        score_weight: float = 0.1,
        **kwargs
    ):
        """
        Initialize advanced Elo model.
        
        Args:
            score_weight: Weight for score differential in rating updates
            **kwargs: Parameters for base EloModel
        """
        super().__init__(**kwargs)
        self.score_weight = score_weight
        self.params["score_weight"] = score_weight
    
    def _update_ratings(self, games: pd.DataFrame) -> None:
        """Update ratings with score differential consideration."""
        # Call parent method first
        super()._update_ratings(games)
        
        # Additional adjustment based on score differential
        for idx, game in games.iterrows():
            if "HS" in game and "AS" in game:
                home_id = game["HID"]
                away_id = game["AID"]
                score_diff = game["HS"] - game["AS"]
                
                # Adjust ratings based on score differential
                adjustment = score_diff * self.score_weight
                self.ratings[home_id] += adjustment
                self.ratings[away_id] -= adjustment


# For backward compatibility, create a Model class that uses EloModel
class Model(EloModel):
    """Default model for submission - uses Elo-based betting."""
    pass
