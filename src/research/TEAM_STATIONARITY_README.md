# Team Strength Stationarity

## Goal
Determine if "Good Teams stay Good" year-over-year.

## Findings
- **Year-over-Year Correlation**: nan

## Results by Season Pair
 Season  Correlation
   1990          NaN
   1991     0.607482
   1992     0.911089
   1993     0.743860
   1994     0.902137
   1995     0.882344
   1996     0.911322
   1997     0.802109
   1998     0.728442
   1999     0.869021
   2000     0.615569
   2001     0.764373
   2002          NaN
   2003          NaN
   2004          NaN
   2005          NaN
   2006          NaN
   2007          NaN
   2008          NaN
   2009          NaN
   2010          NaN

## Implication
- High correlation (> 0.5) means we can rely on last year's data (Model `start_season` can include old data).
- Low correlation means roster turnover is high, and looking back too far adds noise.
