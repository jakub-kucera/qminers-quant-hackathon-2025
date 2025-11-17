# Model Hyperparameters Guide

This document describes all tunable hyperparameters in the ice hockey betting model.

## Overview

The model has been parametrized to allow tuning of key hyperparameters across all sub-models. Parameters were selected based on their expected impact on model performance and their interpretability.

## Ensemble Parameters

### `ensamble_required_n` (default: 500)
- **What**: Minimum number of predictions before the ensemble starts training
- **Range**: 300-1000
- **Impact**: Lower values = faster start but less stable, higher = more stable but slower
- **Tuning tip**: Balance between cold start and reliability

### `ensamble_retrain_freq` (default: 400)
- **What**: How often (in games) to retrain the ensemble logistic regression
- **Range**: 200-800
- **Impact**: Lower = adapts faster to changes, higher = more stable predictions
- **Tuning tip**: Consider seasonality and meta-game evolution

## Sub-Model Parameters

### Pythagorean Model

#### `pythagorean_power` (default: 1.0)
- **What**: Exponent for Pythagorean expectation formula
- **Range**: 0.8-1.5
- **Impact**: Controls non-linearity in win probability calculation
- **Why tunable**: Optimal exponent varies by era and league structure
- **Research**: Linear (1.0) works well for hockey; 1.8-2.0 adds slight non-linearity

#### `pythagorean_regularization` (default: 20)
- **What**: Regularization term added to goals for/against
- **Range**: 10-40
- **Impact**: Higher = more regression to mean for teams with few games
- **Why tunable**: Balances historical vs current performance

#### `pythagorean_decay` (default: 0.992)
- **What**: Daily decay factor for time-weighted statistics
- **Range**: 0.985-0.998
- **Impact**: Lower = more recent games weighted, higher = longer memory
- **Why tunable**: Optimal memory length varies by sport dynamics

### Four Factor Model

#### `four_factor_required_n` (default: 500)
- **What**: Minimum games before Four Factor model makes predictions
- **Range**: 300-1000
- **Impact**: Cold start vs stability trade-off
- **Why tunable**: Balance model reliability with coverage

#### `four_factor_decay` (default: 0.994)
- **What**: Daily decay factor for hockey statistics
- **Range**: 0.990-0.998
- **Impact**: How quickly older game statistics fade
- **Why tunable**: Team form changes vary in persistence

### Gradient Descent

#### `gradient_descent_epochs` (default: 12)
- **What**: Number of training epochs per model fit
- **Range**: 8-20
- **Impact**: Lower = faster but less converged, higher = better fit but slower
- **Why tunable**: Trade-off between computation and accuracy

#### `gradient_descent_min_games` (default: 500)
- **What**: Minimum games before making predictions
- **Range**: 300-800
- **Impact**: Cold start vs reliability
- **Why tunable**: Complex model needs sufficient data

### Exhaustion Model

#### `exhaustion_b2b_penalty` (default: 1.5)
- **What**: Penalty factor for back-to-back games (0 days rest)
- **Range**: 1.0-2.5
- **Impact**: How much to penalize teams on back-to-backs
- **Why tunable**: B2B impact varies by era, team depth, and travel schedule
- **Research**: NHL teams historically perform ~5-10% worse on B2Bs

#### `exhaustion_road_penalty` (default: 0.5)
- **What**: Penalty for long road trips (3+ consecutive away games)
- **Range**: 0.3-0.8
- **Impact**: How much to penalize road-weary teams
- **Why tunable**: Travel fatigue impact varies by geography and scheduling

### ELO System

#### `elo_k_base` (default: 15)
- **What**: Base K-factor for ELO rating updates
- **Range**: 10-25
- **Impact**: Lower = more stable ratings, higher = faster adaptation
- **Why tunable**: Optimal K varies by sport variance and season length
- **Research**: Hockey uses 35-40 (lower than basketball's 50-60) due to randomness

#### `elo_home_advantage` (default: 35)
- **What**: Home ice advantage in ELO points
- **Range**: 25-50
- **Impact**: How much home teams are favored
- **Why tunable**: Home advantage varies by era and league
- **Research**: NHL home win rate ~55% (lower than NBA's 60%)

#### `elo_margin_cap` (default: 3)
- **What**: Maximum goal differential to consider in ELO updates
- **Range**: 2-5
- **Impact**: Prevents blowout/garbage-time goals from skewing ratings
- **Why tunable**: Balance between signal and noise in large victories

### Poisson Goal Model

#### `poisson_regularization` (default: 3.0)
- **What**: Regularization for goal rate estimates
- **Range**: 2.0-5.0
- **Impact**: Higher = more regression to league average
- **Why tunable**: Balance between team-specific and league-average rates

#### `poisson_decay` (default: 0.992)
- **What**: Daily decay for goal rate time-weighting
- **Range**: 0.985-0.998
- **Impact**: How quickly goal-scoring form changes
- **Why tunable**: Offensive/defensive form persistence varies

## Parameters NOT Included

Some parameters were intentionally **not** made tunable:

### Fixed by Design
- **One-day rest penalty** (0.8): Derived as ~53% of B2B penalty - keeps relationship consistent
- **Road trip 2-game penalty** (0.3): Set as 60% of 3+ game penalty - maintains progression
- **Season carryover** (0.75): Standard regression to mean between seasons
- **OT/SO multiplier** (0.7): Based on research that OT games indicate even matchups

### Too Granular
- Number of teams tracked in GradientDescent (50): NHL-specific
- Time window lookbacks (100 games, 20 games): Domain-optimized
- Sample weight decay (0.0003): Ensemble-specific, stable across configs
- LogisticRegression max_iter (10000): Sufficient for convergence

### Architectural
- Sub-model selection: All 6 models provide complementary signals
- Feature engineering: Hockey-specific metrics are domain-driven

## Recommended Tuning Strategy

### Phase 1: Coarse Grid Search
Start with most impactful parameters:
```python
param_grid = {
    'pythagorean_power': [0.8, 1.0, 1.2],
    'elo_k_base': [12, 15, 18],
    'exhaustion_b2b_penalty': [1.2, 1.5, 1.8],
    'ensamble_required_n': [300, 500, 700],
}
```

### Phase 2: Fine-Tuning
Refine promising regions:
```python
param_grid = {
    'pythagorean_power': [0.95, 1.0, 1.05],
    'elo_home_advantage': [30, 35, 40],
    'four_factor_decay': [0.992, 0.994, 0.996],
}
```

### Phase 3: Bayesian Optimization
Use Optuna for final optimization across all parameters simultaneously.

## Usage Examples

### Default Parameters
```python
from src.model import Model

# Use defaults (researched values)
model = Model()
```

### Custom Parameters
```python
# More aggressive adaptation to recent form
model = Model(
    pythagorean_power=1.2,
    pythagorean_decay=0.990,  # Shorter memory
    elo_k_base=18,  # Faster updates
    exhaustion_b2b_penalty=2.0  # Stronger penalty
)
```

### Conservative Parameters
```python
# More stable, long-term focused
model = Model(
    ensamble_required_n=800,  # More training data
    pythagorean_decay=0.996,  # Longer memory
    elo_k_base=12,  # Slower updates
    four_factor_required_n=700
)
```

## Expected Ranges

Based on hockey analytics research:

- **Prediction Accuracy**: 55-62% (vs 50% baseline)
- **Log Loss**: 0.50-0.65 (vs market's ~0.55)
- **Brier Score**: 0.20-0.25 (lower is better)
- **Calibration Error**: < 0.05 (well-calibrated)

## Notes

- All parameters use sensible defaults based on NHL research
- Parameters are independent enough for grid search
- Use cross-validation to avoid overfitting
- Consider computational cost when tuning gradient descent parameters
- Hockey's high variance means small accuracy gains are meaningful
