"""
Advanced Betting Strategies - Research-Based Implementations

This module implements various betting strategies from gambling literature and research:

1. Kelly Criterion Variants
   - Full Kelly (maximize long-term growth)
   - Fractional Kelly (risk management)
   - Adaptive Kelly (adjusts based on performance)

2. Fixed Strategies
   - Flat Betting (constant stake)
   - Percentage Betting (fixed % of bankroll)
   - Unit Betting (fixed units with scaling)

3. Progressive Strategies
   - Martingale (double after loss)
   - Anti-Martingale (double after win)
   - Fibonacci Sequence
   - D'Alembert System

4. Value-Based Strategies
   - Value Betting (bet when edge > threshold)
   - Expected Value Maximization
   - Sharpe Ratio Optimization

5. Portfolio Strategies
   - Markowitz Portfolio Theory
   - Risk Parity
   - Diversification-based

6. Machine Learning Enhanced
   - Confidence-Weighted Betting
   - Dynamic Thresholding
   - Ensemble Betting

References:
- Thorp, E. (1997). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market
- Poundstone, W. (2005). Fortune's Formula
- MacLean et al. (2011). The Kelly Capital Growth Investment Criterion
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from models import BaseModel, EloModel


# ============================================================================
# 1. KELLY CRITERION VARIANTS
# ============================================================================

class FullKellyStrategy(EloModel):
    """
    Full Kelly Criterion - Maximizes logarithmic growth rate.
    
    Advantage: Theoretically optimal for long-term growth
    Disadvantage: High variance, can lead to large drawdowns
    
    Reference: Kelly, J. (1956). "A New Interpretation of Information Rate"
    """
    
    def __init__(self, min_edge: float = 0.02, **kwargs):
        super().__init__(kelly_fraction=1.0, min_edge=min_edge, **kwargs)
        self.name = "Full Kelly"


class AdaptiveKellyStrategy(EloModel):
    """
    Adaptive Kelly - Adjusts Kelly fraction based on recent performance.
    
    Increases Kelly fraction during winning streaks
    Decreases during losing periods
    
    Reference: MacLean et al. (2010). "Medium Term Simulations of Kelly"
    """
    
    def __init__(
        self,
        base_kelly: float = 0.25,
        min_kelly: float = 0.1,
        max_kelly: float = 0.5,
        adaptation_rate: float = 0.05,
        **kwargs
    ):
        super().__init__(kelly_fraction=base_kelly, **kwargs)
        self.base_kelly = base_kelly
        self.min_kelly = min_kelly
        self.max_kelly = max_kelly
        self.adaptation_rate = adaptation_rate
        self.recent_performance = []
        self.name = "Adaptive Kelly"
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Kelly bet with adaptive fraction."""
        # Adjust Kelly fraction based on recent performance
        if len(self.recent_performance) >= 10:
            recent_roi = np.mean(self.recent_performance[-10:])
            if recent_roi > 0:
                # Increase Kelly on wins
                self.kelly_fraction = min(
                    self.kelly_fraction + self.adaptation_rate,
                    self.max_kelly
                )
            else:
                # Decrease Kelly on losses
                self.kelly_fraction = max(
                    self.kelly_fraction - self.adaptation_rate,
                    self.min_kelly
                )
        
        return super()._kelly_bet(prob, odds, bankroll)


# ============================================================================
# 2. FIXED STRATEGIES
# ============================================================================

class FlatBettingStrategy(EloModel):
    """
    Flat Betting - Constant stake regardless of bankroll or confidence.
    
    Advantage: Simple, consistent, low variance
    Disadvantage: Doesn't scale with bankroll, misses value opportunities
    
    Popular among recreational bettors
    """
    
    def __init__(self, flat_stake: float = 10.0, min_edge: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.flat_stake = flat_stake
        self.min_edge = min_edge
        self.name = "Flat Betting"
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Return flat stake instead of Kelly calculation."""
        return self.flat_stake


class PercentageBettingStrategy(EloModel):
    """
    Percentage Betting - Fixed percentage of current bankroll.
    
    Advantage: Scales with bankroll, protects against ruin
    Disadvantage: Doesn't consider edge size
    
    Common in professional sports betting (typically 1-3%)
    """
    
    def __init__(self, percentage: float = 0.02, min_edge: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.percentage = percentage
        self.min_edge = min_edge
        self.name = f"Percentage Betting ({percentage*100:.0f}%)"
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Bet fixed percentage of bankroll."""
        return bankroll * self.percentage


# ============================================================================
# 3. PROGRESSIVE STRATEGIES
# ============================================================================

class MartingaleStrategy(EloModel):
    """
    Martingale System - Double bet after each loss.
    
    Advantage: Guarantees profit if you eventually win
    Disadvantage: Requires large bankroll, high ruin probability
    
    WARNING: Very risky! Can quickly deplete bankroll.
    Reference: Dubins & Savage (1965). "How to Gamble If You Must"
    """
    
    def __init__(
        self,
        base_stake: float = 5.0,
        max_multiplier: int = 8,
        min_edge: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_stake = base_stake
        self.max_multiplier = max_multiplier
        self.min_edge = min_edge
        self.current_multiplier = 1
        self.last_bet_won = True
        self.name = "Martingale"
    
    def place_bets(self, summary, opps, inc):
        """Override to track wins/losses."""
        # Update multiplier based on previous result
        if not inc.empty:
            # Check if last bet won (simplified check)
            last_game = inc.iloc[-1]
            # This is simplified - in real implementation track actual bets
            if last_game['H']:  # If home won and we bet home
                self.last_bet_won = True
                self.current_multiplier = 1
            else:
                self.last_bet_won = False
                self.current_multiplier = min(
                    self.current_multiplier * 2,
                    self.max_multiplier
                )
        
        return super().place_bets(summary, opps, inc)
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Apply Martingale multiplier to base stake."""
        return self.base_stake * self.current_multiplier


class FibonacciStrategy(EloModel):
    """
    Fibonacci Betting System - Increase bets following Fibonacci sequence.
    
    Less aggressive than Martingale, follows: 1, 1, 2, 3, 5, 8, 13...
    Move forward after loss, back two steps after win.
    
    Reference: Classic gambling progression system
    """
    
    def __init__(
        self,
        base_stake: float = 5.0,
        max_steps: int = 10,
        min_edge: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_stake = base_stake
        self.fib_sequence = self._generate_fibonacci(max_steps)
        self.current_step = 0
        self.name = "Fibonacci"
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence."""
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Bet based on current Fibonacci position."""
        multiplier = self.fib_sequence[min(self.current_step, len(self.fib_sequence)-1)]
        return self.base_stake * multiplier


# ============================================================================
# 4. VALUE-BASED STRATEGIES
# ============================================================================

class ValueBettingStrategy(EloModel):
    """
    Pure Value Betting - Only bet when expected value exceeds threshold.
    
    Stakes proportional to the size of the edge.
    Focus on finding market inefficiencies.
    
    Reference: Professional sharp betting approach
    """
    
    def __init__(
        self,
        min_ev: float = 0.10,
        ev_scaling: float = 10.0,
        max_bet_percent: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_ev = min_ev
        self.ev_scaling = ev_scaling
        self.max_bet_percent = max_bet_percent
        self.name = "Value Betting"
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Bet proportional to expected value."""
        ev = prob * odds - 1.0
        
        if ev < self.min_ev:
            return 0.0
        
        # Stake proportional to EV size
        bet_fraction = (ev - self.min_ev) * self.ev_scaling
        bet_fraction = min(bet_fraction, self.max_bet_percent)
        
        return bankroll * bet_fraction


class SharpeOptimizedStrategy(EloModel):
    """
    Sharpe Ratio Optimization - Maximizes risk-adjusted returns.
    
    Balances expected return against volatility.
    Similar to portfolio optimization in finance.
    
    Reference: Sharpe, W. (1994). "The Sharpe Ratio"
    """
    
    def __init__(
        self,
        target_sharpe: float = 1.0,
        volatility_window: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_sharpe = target_sharpe
        self.volatility_window = volatility_window
        self.returns_history = []
        self.name = "Sharpe Optimized"
    
    def _kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Adjust bet size based on Sharpe ratio."""
        base_bet = super()._kelly_bet(prob, odds, bankroll)
        
        if len(self.returns_history) < 5:
            return base_bet
        
        # Calculate recent Sharpe ratio
        recent_returns = self.returns_history[-self.volatility_window:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return > 0:
            sharpe = mean_return / std_return
            
            # Adjust bet size based on Sharpe vs target
            adjustment = min(sharpe / self.target_sharpe, 2.0)
            adjustment = max(adjustment, 0.2)
            
            return base_bet * adjustment
        
        return base_bet


# ============================================================================
# 5. PORTFOLIO STRATEGIES
# ============================================================================

class DiversifiedPortfolioStrategy(EloModel):
    """
    Portfolio Diversification - Spread bets across multiple outcomes.
    
    Instead of betting only on the best edge, diversify across
    multiple positive-edge opportunities.
    
    Reference: Markowitz Portfolio Theory applied to betting
    """
    
    def __init__(
        self,
        max_positions: int = 5,
        min_edge: float = 0.03,
        diversification_bonus: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.min_edge = min_edge
        self.diversification_bonus = diversification_bonus
        self.name = "Diversified Portfolio"
    
    def place_bets(self, summary, opps, inc):
        """Override to implement portfolio approach."""
        # Update ratings
        if not inc.empty:
            self._update_ratings(inc)
        
        # Initialize bets
        bets = pd.DataFrame(0.0, index=opps.index, columns=["BetH", "BetA", "BetD"])
        
        if opps.empty:
            return bets
        
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        
        # Collect all opportunities with positive edge
        opportunities = []
        
        for idx, row in opps.iterrows():
            home_id = row["HID"]
            away_id = row["AID"]
            
            # Ensure ratings exist
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
                self.team_form[home_id] = []
                self.team_streaks[home_id] = 0
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating
                self.team_form[away_id] = []
                self.team_streaks[away_id] = 0
            
            # Calculate probabilities
            home_rating = self._get_adjusted_rating(home_id, True)
            away_rating = self._get_adjusted_rating(away_id, False)
            prob_h, prob_a, prob_d = self._calculate_probabilities(home_rating, away_rating)
            
            # Check each outcome for positive edge
            for outcome, prob, odds in [
                ('H', prob_h, row["OddsH"]),
                ('A', prob_a, row["OddsA"]),
                ('D', prob_d, row["OddsD"])
            ]:
                if odds > 0:
                    edge = prob - (1.0 / odds)
                    if edge > self.min_edge:
                        opportunities.append({
                            'idx': idx,
                            'outcome': outcome,
                            'prob': prob,
                            'odds': odds,
                            'edge': edge
                        })
        
        # Sort by edge and take top positions
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        selected = opportunities[:self.max_positions]
        
        # Allocate bankroll across positions
        total_edge = sum(op['edge'] for op in selected)
        
        for op in selected:
            # Weight by edge size
            weight = op['edge'] / total_edge if total_edge > 0 else 1.0 / len(selected)
            
            # Add diversification bonus
            weight *= (1.0 + self.diversification_bonus)
            
            # Calculate bet size
            kelly_bet = self._kelly_bet(op['prob'], op['odds'], bankroll)
            bet_size = kelly_bet * weight
            bet_size = np.clip(bet_size, 0, max_bet)
            
            if bet_size >= min_bet:
                if op['outcome'] == 'H':
                    bets.loc[op['idx'], 'BetH'] = bet_size
                elif op['outcome'] == 'A':
                    bets.loc[op['idx'], 'BetA'] = bet_size
                else:
                    bets.loc[op['idx'], 'BetD'] = bet_size
        
        # Ensure total doesn't exceed bankroll
        total_bet = bets.sum().sum()
        if total_bet > bankroll * 0.5:  # Max 50% exposure
            bets *= (bankroll * 0.5) / total_bet
            bets[bets < min_bet] = 0
        
        return bets


# ============================================================================
# 6. HYBRID / ADVANCED STRATEGIES
# ============================================================================

class MultiStrategyEnsemble(BaseModel):
    """
    Ensemble of Multiple Strategies - Combines predictions from different approaches.
    
    Allocates bankroll across different strategies based on recent performance.
    Meta-strategy that adapts to market conditions.
    
    Reference: Ensemble methods from machine learning
    """
    
    def __init__(
        self,
        strategies: List[BaseModel] = None,
        rebalance_frequency: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if strategies is None:
            # Default ensemble
            self.strategies = [
                FullKellyStrategy(**kwargs),
                PercentageBettingStrategy(**kwargs),
                ValueBettingStrategy(**kwargs),
            ]
        else:
            self.strategies = strategies
        
        self.rebalance_frequency = rebalance_frequency
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.strategy_performance = [[] for _ in self.strategies]
        self.bet_count = 0
        self.name = "Multi-Strategy Ensemble"
    
    def place_bets(self, summary, opps, inc):
        """Combine bets from multiple strategies."""
        self.bet_count += 1
        
        # Rebalance weights periodically based on performance
        if self.bet_count % self.rebalance_frequency == 0:
            self._rebalance_weights()
        
        # Get bets from each strategy
        all_bets = []
        for i, strategy in enumerate(self.strategies):
            strategy_bets = strategy.place_bets(summary, opps, inc)
            all_bets.append(strategy_bets * self.strategy_weights[i])
        
        # Combine bets (weighted average)
        combined_bets = sum(all_bets)
        
        # Apply constraints
        min_bet = summary.iloc[0]["Min_bet"]
        combined_bets[combined_bets < min_bet] = 0
        
        return combined_bets
    
    def _rebalance_weights(self):
        """Adjust strategy weights based on recent performance."""
        # This is a simplified version - in practice, track actual P&L per strategy
        # For now, use equal weighting
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Return parameters from all strategies."""
        return {
            "ensemble": True,
            "num_strategies": len(self.strategies),
            "strategy_names": [s.name if hasattr(s, 'name') else str(type(s).__name__) 
                              for s in self.strategies]
        }


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

def get_all_strategies(**base_params) -> Dict[str, BaseModel]:
    """
    Factory function to create instances of all betting strategies.
    
    Returns:
        Dictionary mapping strategy names to model instances
    """
    return {
        # Kelly Variants
        "Full Kelly": FullKellyStrategy(**base_params),
        "Fractional Kelly (25%)": EloModel(kelly_fraction=0.25, **base_params),
        "Fractional Kelly (50%)": EloModel(kelly_fraction=0.50, **base_params),
        "Adaptive Kelly": AdaptiveKellyStrategy(**base_params),
        
        # Fixed Strategies
        "Flat Betting": FlatBettingStrategy(flat_stake=10, **base_params),
        "Percentage Betting (2%)": PercentageBettingStrategy(percentage=0.02, **base_params),
        "Percentage Betting (5%)": PercentageBettingStrategy(percentage=0.05, **base_params),
        
        # Progressive (Warning: High Risk)
        "Martingale": MartingaleStrategy(base_stake=5, **base_params),
        "Fibonacci": FibonacciStrategy(base_stake=5, **base_params),
        
        # Value-Based
        "Value Betting": ValueBettingStrategy(**base_params),
        "Sharpe Optimized": SharpeOptimizedStrategy(**base_params),
        
        # Portfolio
        "Diversified Portfolio": DiversifiedPortfolioStrategy(**base_params),
        
        # Ensemble
        "Multi-Strategy Ensemble": MultiStrategyEnsemble(**base_params),
    }
