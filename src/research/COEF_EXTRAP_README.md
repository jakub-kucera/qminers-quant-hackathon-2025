# Coefficient Extrapolation (Stability)

## Goal
Check if the predictive power of a feature (Previous Goal Difference) assumes a constant value or drifts over time.

## Findings
- **Mean Coefficient**: 0.0449
- **Std Dev**: 0.0350
- **Stability**: See `coef_extrapolation.png`.

## Implication
- If the plot is flat, the game meta is stable.
- If it trends up/down, the importance of "Momentum" is changing.
- Large fluctuations suggest high variance or regime changes (e.g., rule changes).
