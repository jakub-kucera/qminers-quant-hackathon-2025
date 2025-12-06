# Head-to-Head (H2H) Tendencies

## Goal
Identify "Bogey Teams" or matchups where one team consistently overperforms the odds.

## Findings
filtered for Matchups with >= 10 games.

### Top Home Overperformers (Home Team wins more than expected)
```
 Matchup  Games  Avg_Bias
(12, 16)     12       NaN
(10, 19)     87       NaN
 (0, 13)     16       NaN
(22, 24)     84       NaN
  (4, 5)     28       NaN
```

### Top Away Overperformers (Home Team underperforms)
```
 Matchup  Games  Avg_Bias
 (2, 20)     16 -0.292922
(11, 15)     20 -0.261929
(11, 18)     20 -0.256493
 (4, 18)     22 -0.252774
(14, 23)     12 -0.251624
```

## Implication
- If significant biases exist (e.g., > 10-20% edge), adding an `H2H_Bias` feature or `MatchupID` embedding is valuable.
- Caution: Small sample sizes (even 10 games is small for variance).
