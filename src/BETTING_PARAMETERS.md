# Betting Strategy Parameters Guide

This document describes the betting strategy hyperparameters that control how the model places bets based on its predictions.

## Overview

After optimizing prediction accuracy with model hyperparameters, betting strategy parameters control:
1. **When to bet**: Minimum edge required over market odds
2. **How much to bet**: Base bet sizing and Kelly criterion application

These parameters are separate from prediction hyperparameters to allow independent optimization.

## Betting Strategy Parameters

### `bet_threshold` (default: 0.05)
- **What**: Minimum edge over implied odds required to place a bet
- **Range**: 0.0-0.15
- **Formula**: `edge = predicted_probability - implied_probability`
- **Impact**: 
  - Lower = more bets placed (higher volume, higher risk)
  - Higher = fewer bets placed (lower volume, more selective)
- **Example**: 
  - threshold=0.05: Only bet when predicted prob is 5% better than market
  - threshold=0.10: Only bet when predicted prob is 10% better than market
- **Tuning tip**: Higher thresholds are more conservative but may miss value

### `bet_size_base` (default: 0.02)
- **What**: Base bet size as a fraction of bankroll
- **Range**: 0.005-0.05 (0.5% to 5% of bankroll)
- **Impact**: 
  - Lower = more conservative, lower variance
  - Higher = more aggressive, higher variance
- **Cap**: Maximum bet size is capped at 3x this value
- **Example**:
  - bet_size_base=0.02: Base bet is 2% of bankroll
  - bet_size_base=0.04: Base bet is 4% of bankroll
- **Tuning tip**: Consider bankroll volatility and risk tolerance

### `bet_kelly_fraction` (default: 0.25)
- **What**: Fraction of Kelly criterion to use for bet sizing
- **Range**: 0.1-0.5 (fractional Kelly)
- **Formula**: `kelly_size = kelly_fraction * (edge / (odds - 1))`
- **Impact**: 
  - Lower = more conservative (0.25 = quarter Kelly)
  - Higher = more aggressive (0.5 = half Kelly)
- **Why fractional**: Full Kelly is optimal for log wealth but too aggressive for most bettors
- **Research**: Quarter to half Kelly is commonly recommended for practical betting
- **Tuning tip**: Lower fractions provide better risk-adjusted returns

## Betting Logic

The model uses the following logic to determine bets:

```python
# 1. Calculate edge over implied odds
implied_home = 1 / odds_home
edge_home = predicted_prob - implied_home

# 2. Check if edge exceeds threshold
if edge_home > bet_threshold:
    # 3. Calculate Kelly criterion bet size
    kelly_size = bet_kelly_fraction * (edge_home / (odds_home - 1))
    
    # 4. Apply constraints
    bet_amount = max(min_bet, min(max_bet, bankroll * min(kelly_size, bet_size_base * 3)))
```

## Optimization Strategy

### Sequential Approach (Recommended)
1. **First**: Optimize prediction hyperparameters for accuracy
   - Use `optimize_hyperparameters.py`
   - Maximize prediction accuracy metrics
   
2. **Second**: Optimize betting strategy for bankroll
   - Use `optimize_betting.py`
   - Maximize final bankroll with fixed prediction hyperparameters

### Tools

**Prediction Optimization**:
```bash
python3 src/optimize_hyperparameters.py --n-trials 200 --n-jobs 4
```

**Betting Optimization**:
```bash
python3 src/optimize_betting.py --n-trials 100 --n-jobs 4
```

## Usage Examples

### Conservative Strategy
```python
model = Model(
    bet_threshold=0.10,      # Only bet with 10% edge
    bet_size_base=0.01,      # 1% base bet size
    bet_kelly_fraction=0.20  # One-fifth Kelly
)
```

### Balanced Strategy (Default)
```python
model = Model(
    bet_threshold=0.05,      # Bet with 5% edge
    bet_size_base=0.02,      # 2% base bet size
    bet_kelly_fraction=0.25  # Quarter Kelly
)
```

### Aggressive Strategy
```python
model = Model(
    bet_threshold=0.03,      # Bet with 3% edge
    bet_size_base=0.04,      # 4% base bet size
    bet_kelly_fraction=0.40  # Two-fifths Kelly
)
```

## Expected Results

Based on hockey betting research and market efficiency:

- **Conservative**: Lower variance, more stable bankroll growth, fewer bets
- **Balanced**: Moderate variance, good risk-adjusted returns, reasonable bet frequency
- **Aggressive**: Higher variance, potential for higher returns, more bets, higher risk

## Risk Management

**Key Principles**:
1. Never bet more than necessary to achieve Kelly sizing
2. Cap maximum bet at 3x base size to limit exposure
3. Use fractional Kelly (not full Kelly) to reduce variance
4. Higher thresholds in efficient markets (NHL is relatively efficient)

**Bankroll Considerations**:
- Start conservative and increase aggression as bankroll grows
- Consider using geometric mean (Kelly) rather than expected value
- Monitor actual vs predicted performance regularly

## Relationship to Model Hyperparameters

**Separation of Concerns**:
- **Model hyperparameters** (pythagorean_power, elo_k_base, etc.): Optimize prediction accuracy
- **Betting parameters** (bet_threshold, bet_size_base, etc.): Optimize betting returns

**Why separate?**:
1. Prediction accuracy is objective and measurable
2. Betting strategy depends on risk tolerance and bankroll
3. Same model can support different betting strategies
4. Easier to tune independently

## Further Reading

- **Kelly Criterion**: Original paper by J.L. Kelly Jr. (1956)
- **Fractional Kelly**: Recommendations by Edward Thorp and others
- **Sports Betting Markets**: Efficiency studies in NHL betting markets
- **Bankroll Management**: Professional gambling literature
