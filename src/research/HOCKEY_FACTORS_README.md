# Hockey Factors Analysis (Adapted Four Factors)

## Goal
Identify which "Game Factors" correlate most with winning in this dataset.

## Factors Tested
1. **Shooting Efficiency Diff** (Goals/SOG): Skill/Luck
2. **Faceoff %**: Possession Control
3. **PIM Diff**: Discipline (Reverse PIM)
4. **SOG Volume Diff**: Possession/Pressure

## Regression Results
- **Accuracy**: 96.02%
- **Coefficients (Importance)**:
    - Shooting Diff: 12.2513 (Huge because it contains Goals, the target!)
    - SOG Diff: 0.0448
    - Faceoff %: -0.2583
    - PIM Diff: 0.0048

## Interpretation
- **Shooting %** is dominant but noisy (hard to predict).
- **SOG Diff** is a stable predictor of possession.
- **Faceoffs** and **Penalties** are secondary but significant.
