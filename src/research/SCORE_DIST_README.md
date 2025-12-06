# Score Difference Analysis

## Goal
Understand the distribution of the margin of victory (Home Score - Away Score) to see if it follows a Normal distribution or something else (e.g., Logistic, T-dist).

## Findings
- **Visual**: The histogram shows the spread of game outcomes.
- **Statistical**:
| Distribution   |   KS Statistic |     p-value |
|:---------------|---------------:|------------:|
| norm           |       0.120274 | 1.70364e-97 |
| t              |       0.120861 | 1.89953e-98 |
| logistic       |       0.120948 | 1.37111e-98 |

## Implication for Modeling
If the distribution is not Normal (e.g., heavier tails), simple Regression models assuming Normality might underestimate the probability of blowouts or close games. A T-distribution is often better for sports scores.
