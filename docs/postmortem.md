# Post-mortem

We qualified for the finals with the hybrid model and finished 8th of 20 without placing a bet.
This is what happened in each finals round, what every model would have done there, and what
we would do differently. All numbers come from `tools/replay.py` and `tools/summarise.py` on the
data the organisers published after the event (`results/`).

## Timeline

| When | What |
|---|---|
| 5 Nov 2025 | The task is published: games from 1989/90 to 2010/11, evaluation on the next four seasons. |
| 16-22 Nov | The ensemble, a hockey port of the 2024 winners' design, largely written by Copilot's coding agent; Optuna on its settings. Pepe's staking rule joins it as the hybrid (22 Nov). |
| 17-19 Nov | Pepe's branch: a Copilot pull request with the team's first betting strategies, among them the confidence-weighted Kelly that the hybrid later took. |
| 21-26 Nov | Marek's models on his own branch: XGBoost, then the market-residual network. |
| 25-28 Nov | The Bookie Rookie task (a second bookmaker with looser odds) and the hybrid tuned for it. |
| 4-5 Dec | A deliberately simple model (`simple-rolling`), and analysis scripts. |
| 6 Dec | The finals: three rounds in one day, each scored on two new seasons with a fresh bankroll of 1,000. |

## The finals, round by round

### Round 1 (seasons 2015-16): no bets, by choice

A model's first day in a round replays the whole history, 10,700 to 12,400 games, and the
hybrid refits its margin ratings (a few Adam steps over the last two years of games) for every
new date.
In the replay that costs about 100 CPU-seconds on a laptop; on the day, on the server's one CPU,
training ran longer than we had planned for, so we uploaded a model that bets nothing.

Had the hybrid run, it would have lost 130 on 2 bets (the version uploaded in round 2) or made
18 on 4 (the retuned one).

### Round 2 (seasons 2017-18): the hybrid, no bets

Replayed on round 2, the hybrid predicts every game on every day it is offered, 1,801
predictions in all, and bets none. Pepe's confidence score has to reach 0.61 under these
settings; its median is 0.39, and the 52 predictions that clear it all fall short of the 7.6%
minimum edge the settings also demand. The thresholds, tuned on 2010-14, let nothing through.

Two versions of the hybrid exist from that day. Replayed on round 2, the one committed at 12:04
places no bets, matching what the server reported; the one retuned later places five. We take
the 12:04 one to be what went in (`originals/model_hybrid_round_2.py`); the server keeps the
actual upload.

### Round 3 (seasons 2019-20): Pepe's underdog sniper, no bets on the server

The sniper refits its classifier only when its number of training rows is an exact multiple of
100 at the start of a day. Replayed on round 3, that first happens on day 30 of 412, and only five
times in the round. So whether and when it trains depends on the exact count of games it has
seen: one game more or less in what the server fed it would explain no bets at all.

On the published data it trains, bets 250 twice on each of two away underdogs (odds 3.12 and
3.68) whose model probability cleared the 20% expected-value bar, and loses the whole bankroll.
Not betting was the better outcome.

## What every model would have done

Profit on a bankroll of 1,000, with the number of bets in brackets. The betting window is opened
for every model (`WHAT_IF` in `tools/replay.py`), so `hybrid-retuned` bets from 2015 and
`simple-rolling` past 2018. An asterisk marks a round whose seasons the model's settings were
tuned on. The qualification round is rebuilt from the finals' training data, whose odds for
earlier seasons differ from what qualification used, so treat that column as approximate.

| model | qualification | finals-1 | finals-2 | finals-3 |
|---|---:|---:|---:|---:|
| ensemble | +167 (163) | +113 (36) | -103 (80) | +271 (58) |
| hybrid | +781 (15) * | -130 (2) | +0 (0) | +0 (0) |
| hybrid-retuned | +291 (20) * | +18 (4) | +25 (5) | +542 (4) |
| bookie-rookie | +676 (184) | +161 (54) | -541 (72) | +1,658 (58) |
| simple-rolling | n/a | -999 (51) | +137 (26) * | -660 (20) |
| underdog-sniper | -700 (4) | +0 (0) | -1,000 (2) | -1,000 (2) |
| market-residual | -996 (351) | -1,000 (503) | -999 (381) | -999 (540) |
| no-bets | +0 (0) | +0 (0) | +0 (0) | +0 (0) |

`simple-rolling` learns only from 2015 on, so it has nothing to say about the qualification
seasons.

Pooled over the three finals rounds (`results/finals_summary.csv`). The z-score is the profit
in standard deviations of what fair odds would give; the bookmaker's margin makes even that
generous to the models.

| model | bets | staked | profit | return on stake | z |
|---|---:|---:|---:|---:|---:|
| ensemble | 174 | 2,211 | +280 | +12.7% | 1.56 |
| hybrid | 2 | 327 | -130 | -39.8% | -0.84 |
| hybrid-retuned | 13 | 1,580 | +585 | +37.0% | 1.42 |
| bookie-rookie | 184 | 11,995 | +1,279 | +10.7% | 0.73 |
| simple-rolling | 97 | 20,405 | -1,522 | -7.5% | -0.85 |
| underdog-sniper | 4 | 2,000 | -2,000 | -100% | -1.30 |
| market-residual | 1,424 | 11,689 | -2,998 | -25.6% | -3.72 |

None of the profits is distinguishable from luck. The market-residual network's loss is.

## What we would do differently

1. **Time the bot on one CPU, on the full history, before the day.** Round 1 was lost to a
   training run nobody had timed on the server's hardware.
2. **Tune on one period and score on the next.** Every setting we shipped came from Optuna runs
   scored on the same seasons they were tuned on (`results/optuna_studies.csv`): the round-2
   hybrid's settings made +1,228 on 2010-14 and then -130, 0 and 0 in the finals; Bookie Rookie's
   reported +3,031% in sample. The searches were unstable too: the simple model's best minimum
   edge was 0.13 on round 1's seasons and 0.013 on round 2's.
3. **Count the bets before believing a profit.** The hybrid bets 0 to 5 times a round; its
   +542 in round 3 is four bets.
4. **Check calibration before staking Kelly.** Marek's network adds a correction to the
   bookmaker's log-odds and stakes 30% Kelly on every game where it sees value; with probabilities
   that aren't sharper than the market's, that bets on most games and goes bust in every round.
   Its inputs are divided by the variance instead of the standard deviation, but dividing by the
   standard deviation goes bust too.
5. **Retrain on a schedule that can't be skipped**: every N new games (`>=`), never when a count
   `% N == 0`.
6. **Expect the data to change shape between rounds.** In the finals files the history's odds are
   set to zero, its rows reordered within each day and renumbered. In the qualification data, the
   seasons from 2005 have no draw odds, and the ensemble and the hybrid skip every game whose draw
   odds are missing; the finals data has them.
7. **Keep the exact uploads.** We can't tell for sure which hybrid went into round 2.

## Known flaws, kept as submitted

`qqh/` reproduces the originals bet for bet, flaws included:

- `underdog-sniper`: training rows are built after the game has been added to the signals, so
  they see the result; the refit condition above; one isotonic calibrator shared by home and
  away probabilities.
- `market-residual`: the variance scaling; the correction is added to the standardised
  log-odds, not the raw ones; "days since the last game" only ever sees the last away game;
  "Corsi" and "Fenwick" count the opponent's shots on goal and the saves rather than shot
  attempts. The training script isn't in the repository, only the trained weights.
- `simple-rolling` bets on home wins only; `PoissonGoals` ignores its `regularization`.
