"""Pool each model's finals bets (results/bets.csv) and test them against luck.

    uv run python tools/summarise.py

The bets file keeps only the stakes; their odds and results are read from the organisers'
data (scripts/fetch_data.py).

If the bookmaker's odds were fair, a stake s at decimal odds o would win s(o - 1) with
probability 1/o and lose s otherwise: mean 0, variance s^2 (o - 1). The z-score is the total
profit over the square root of the summed variances. The bookmaker's margin makes the true
null mean negative, so a positive z overstates the evidence for skill.

Writes results/finals_summary.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from qqh.environment import ODDS_COLS, RESULT_COLS
from qqh.models import MODELS
from qqh.rounds import ROUNDS

RESULTS = Path(__file__).resolve().parent.parent / "results"
FINALS = ["finals-1", "finals-2", "finals-3"]
SIDES = [("BetH", "OddsH", "H"), ("BetA", "OddsA", "A"), ("BetD", "OddsD", "D")]


def pool(bets: pd.DataFrame) -> dict:
    stakes, profits, variances = [], [], []
    for stake_col, odds_col, result_col in SIDES:
        placed = bets[bets[stake_col] > 0]
        stake = placed[stake_col].to_numpy()
        odds = placed[odds_col].to_numpy()
        won = placed[result_col].astype(bool).to_numpy()
        stakes.append(stake)
        profits.append(np.where(won, stake * (odds - 1), -stake))
        variances.append(stake**2 * (odds - 1))
    stake, profit, variance = (np.concatenate(x) for x in (stakes, profits, variances))
    return {
        "bets": len(stake),
        "staked": round(stake.sum(), 2),
        "profit": round(profit.sum(), 2),
        "roi_on_stake": round(profit.sum() / stake.sum(), 3) if len(stake) else None,
        "z": round(profit.sum() / np.sqrt(variance.sum()), 2) if len(stake) else None,
    }


def main() -> None:
    bets = pd.read_csv(RESULTS / "bets.csv", float_precision="round_trip")
    games = {name: ROUNDS[name].games()[[*ODDS_COLS, *RESULT_COLS]] for name in FINALS}
    rows = []
    for name in MODELS:
        mine = bets[(bets["model"] == name) & bets["round"].isin(FINALS)]
        frames = [part.join(games[r], on="game") for r, part in mine.groupby("round")]
        columns = [c for side in SIDES for c in side]
        placed = pd.concat(frames) if frames else pd.DataFrame(columns=columns)
        rows.append({"model": name, **pool(placed)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "finals_summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
