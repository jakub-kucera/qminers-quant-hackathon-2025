# Vegas Odds Accuracy Analysis

## Goal
Determine how accurate the closing odds are. This sets a "baseline" for our models. It is very hard to beat Vegas.

## Results
- **MSE (Brier Score)**: 0.40308 (Lower is better)
- **Log Loss**: 9.75537
- **Baseline Accuracy**: 50.58% (If you just bet on the favorite)

## Calibration
How well do the odds match reality?
```
 Predicted   Actual  Count
     0.025 0.616794   3275
     0.175 0.400000      5
     0.225 0.066667     30
     0.275 0.337662     77
     0.325 0.335526    152
     0.375 0.439189    296
     0.425 0.420940    468
     0.475 0.490765    758
     0.525 0.550321    934
     0.575 0.627365    687
     0.625 0.712919    418
     0.675 0.763265    245
     0.725 0.764706     68
     0.775 0.800000     20
     0.825 1.000000      1
```

## Implication
Our simplified model needs to approach this MSE to be competitive. If our MSE is significantly higher, we are missing information that Vegas has.
