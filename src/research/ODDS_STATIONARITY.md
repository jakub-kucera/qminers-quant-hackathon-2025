# Odds Stationarity Analysis (ADF Test)

## Goal
Check if the market efficiency (or error) is "stationary" over time.
Stationarity means the statistical properties (mean error, variance) do not change over time.

## Results
- **ADF Statistic**: -47.8910
- **p-value**: 0.0000e+00
- **Stationary?**: YES

## Interpretation
Since p-value < 0.05, we **Reject** the Null Hypothesis.
- **Stationary (Yes)**: The market's error rate is stable. We can model it with a time-invariant model (like simple rolling averages).
- **Non-Stationary (No)**: The market is evolving (e.g., becoming more efficient). We might need models that adapt quickly (like recent-window only).
