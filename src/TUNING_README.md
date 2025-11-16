# Model Tuning Framework

This directory contains tools for tuning and evaluating the ice hockey betting model.

## Overview

The framework separates **prediction accuracy** optimization from **betting strategy** optimization, allowing you to focus on model quality first.

## Files

### 1. `quick_eval.py` - Fast Model Evaluation

Quick assessment of model prediction accuracy without full betting simulation.

**Usage:**
```bash
cd /home/runner/work/QQH-2025/QQH-2025
python3 src/quick_eval.py
```

**Features:**
- Evaluates prediction accuracy vs market odds
- Calculates log-loss, Brier score, calibration error
- Compares across different time periods
- Fast feedback loop for development

**Output:**
- Accuracy metrics (model vs market)
- Calibration analysis
- Hypothetical ROI estimates

### 2. `tune_model.py` - Cross-Validation Framework

Comprehensive model evaluation with time-based cross-validation.

**Usage:**
```bash
python3 src/tune_model.py
```

**Features:**
- Time-based train/validation splits (no data leakage)
- K-fold cross-validation
- Grid search over hyperparameters
- Detailed metrics tracking
- Test set evaluation

**Key Functions:**
- `create_time_based_splits()`: Creates chronological CV folds
- `cross_validate()`: Evaluates model with CV
- `grid_search()`: Systematic hyperparameter search
- `evaluate_on_test_set()`: Final holdout evaluation

**Example:**
```python
from src.tune_model import ModelTuner

tuner = ModelTuner()

# Define parameter grid
param_grid = {
    'ensamble_required_n': [300, 500, 700],
    'pythagorean_power': [0.8, 1.0, 1.2],
}

# Search
best_params, results = tuner.grid_search(Model, param_grid, n_folds=3)

# Final test
test_metrics = tuner.evaluate_on_test_set(Model, best_params)
```

### 3. `optimize_hyperparameters.py` - Optuna Optimization

Advanced hyperparameter optimization using Bayesian methods.

**Usage:**
```bash
python3 src/optimize_hyperparameters.py
```

**Features:**
- Bayesian optimization (more efficient than grid search)
- Automatic hyperparameter suggestion
- Progress tracking
- Saves best parameters

**Requirements:**
```bash
pip install optuna
```

**Example:**
```python
from src.optimize_hyperparameters import ModelOptimizer

optimizer = ModelOptimizer(validation_period=365)
best_params = optimizer.optimize(n_trials=100)
```

## Workflow

### Phase 1: Quick Iteration
Use `quick_eval.py` for rapid testing during development:
```bash
# Make changes to model.py
python3 src/quick_eval.py
# Review metrics, iterate
```

### Phase 2: Hyperparameter Tuning

#### Option A: Grid Search (exhaustive)
```python
from src.tune_model import ModelTuner

tuner = ModelTuner()
param_grid = {
    'ensamble_required_n': [300, 500, 700, 1000],
    # Add more parameters
}
best_params, results = tuner.grid_search(Model, param_grid, n_folds=3)
```

#### Option B: Bayesian Optimization (recommended)
```bash
python3 src/optimize_hyperparameters.py
```

### Phase 3: Final Evaluation
```python
from src.tune_model import ModelTuner

tuner = ModelTuner()
test_metrics = tuner.evaluate_on_test_set(Model, best_params)
```

### Phase 4: Betting Strategy Optimization
After model accuracy is good, optimize betting parameters:
- Bet sizing
- Confidence thresholds
- Kelly criterion fractions

## Metrics Explained

### Accuracy
Binary classification accuracy: % of correct predictions (home win vs not)

### Log Loss (Cross-Entropy)
Measures probability calibration. Lower is better.
- Perfect predictions: 0.0
- Random (50/50): 0.693
- Market odds are typically ~0.50-0.55

### Brier Score
Mean squared error of probabilities. Lower is better.
- Perfect: 0.0
- Random: 0.25

### Calibration Error
How well predicted probabilities match actual frequencies.
- Lower is better
- < 0.05 is well-calibrated

## Adding Tunable Parameters

To make model parameters tunable:

1. **Modify `Model.__init__()` to accept parameters:**
```python
class Model:
    def __init__(self, ensamble_required_n=500, pythagorean_power=1.0):
        self.ensamble_required_n = ensamble_required_n
        self.model_list = [
            Pythagorean(power=pythagorean_power),
            # ...
        ]
```

2. **Add to parameter grids:**
```python
param_grid = {
    'ensamble_required_n': [300, 500, 700],
    'pythagorean_power': [0.8, 1.0, 1.2, 1.5],
}
```

3. **Update optimizer objective:**
```python
def objective(self, trial):
    params = {
        'ensamble_required_n': trial.suggest_int('ensamble_required_n', 300, 1000),
        'pythagorean_power': trial.suggest_float('pythagorean_power', 0.8, 1.5),
    }
    # ...
```

## Tips

1. **Start Simple**: Use `quick_eval.py` first to verify model runs
2. **Validate Properly**: Always use time-based splits (no shuffling!)
3. **Focus on Log Loss**: Better metric than accuracy for probability predictions
4. **Check Calibration**: Well-calibrated probabilities are crucial for betting
5. **Beat the Market**: Compare against market odds (implied probabilities)
6. **Iterate**: Model tuning is iterative - don't expect perfection immediately

## Expected Performance

Based on hockey analytics research:
- **Accuracy ceiling**: ~59-62% (hockey is high variance)
- **Target log loss**: < 0.60 (better than market's ~0.55)
- **Calibration error**: < 0.05 (well-calibrated)

## Output Files

- `tuning_results.json`: Grid search results
- `optimization_results.json`: Optuna optimization results
- `test_predictions.csv`: Final test set predictions
- Individual trial data for analysis

## Next Steps

After achieving good prediction accuracy:
1. Implement betting strategy optimization
2. Add Kelly criterion sizing
3. Optimize confidence thresholds
4. Consider ensemble betting strategies
5. Add risk management rules
