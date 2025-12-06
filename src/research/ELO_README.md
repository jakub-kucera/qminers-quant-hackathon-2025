# ELO Simulation Analysis

## Goal
Verify if a simple ELO model has predictive power on this specific dataset (`games.csv`).

## Method
- Simulated a standard ELO system (K=6) over the entire history.
- **Sequential**: Training happens game-by-game.

## Results
- **MSE**: 0.24137
- **Accuracy**: 57.29%

## Conclusion
A pure ELO models yields ~57.3% accuracy. This serves as a "Zero-Feature" baseline. The Logistic/Neural models must beat this significantly to verify that other features (SOG, etc.) are adding value.
