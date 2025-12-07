"""The evaluation loop: a day-by-day replay of a games table against a betting model.

It follows the organisers' rules (their `src/environment.py`, which `scripts/fetch_data.py`
downloads to `data/reference/` and `tests/test_environment.py` checks this against):

- Each day the model gets the results it hasn't seen (`inc`), the games it can bet on
  (`opps`: odds open, game not yet played, odds non-zero) and its bankroll (`summary`).
- Bets below `min_bet` or above `max_bet` are dropped one by one; if the rest add up to more
  than the bankroll, the whole day's bets are dropped.
- Stakes leave the bankroll when placed and winnings (stake x odds) come back the day after
  the game.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

RESULT_COLS = ["H", "A", "D"]
ODDS_COLS = ["OddsH", "OddsA", "OddsD"]
BET_COLS = ["BetH", "BetA", "BetD"]
# Everything a model must not see before the game is played.
FEATURE_COLS = [
    "HS", "AS", "Special", "H_PEN", "A_PEN", "H_MAJ", "A_MAJ", "H_PPG", "A_PPG", "H_SHG",
    "A_SHG", "H_SV", "A_SV", "H_PIM", "A_PIM", "H_SOG", "A_SOG", "H_BLK_S", "A_BLK_S",
    "H_HIT", "A_HIT", "H_BLK", "A_BLK", "H_FO", "A_FO", "H_P1", "A_P1", "H_P2", "A_P2",
    "H_P3", "A_P3", "H_OT", "A_OT", "H_SO", "A_SO",
]  # fmt: skip


class BettingModel(Protocol):
    def place_bets(
        self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame
    ) -> pd.DataFrame: ...


@dataclass
class Result:
    """A finished run: the games with the stakes placed on them, and the bankroll over time."""

    games: pd.DataFrame
    bankroll: float
    history: pd.DataFrame = field(repr=False)

    @property
    def bets(self) -> pd.DataFrame:
        staked = self.games[BET_COLS]
        return self.games.loc[staked.sum(axis=1) > 0]

    @property
    def staked(self) -> float:
        return float(self.games[BET_COLS].values.sum())


class Environment:
    def __init__(
        self,
        games: pd.DataFrame,
        model: BettingModel,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        init_bankroll: float = 1000.0,
        min_bet: float = 0,
        max_bet: float = np.inf,
    ):
        self.games = games.copy()
        self.games[BET_COLS] = 0.0
        self.start_date = start_date if start_date is not None else self.games["Open"].min()
        self.end_date = end_date if end_date is not None else self.games["Date"].max()
        self.model = model
        self.bankroll = init_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.last_seen = pd.Timestamp("1900-01-01")
        self.history: dict[str, list] = {"Date": [], "Bankroll": [], "Cash_Invested": []}

    def run(self) -> Result:
        for date in pd.date_range(self.start_date, self.end_date):
            inc = self._settle(date)
            opps = self._options(date)
            if opps.empty and inc.empty:
                continue
            bets = self.model.place_bets(self._summary(date), opps, inc)
            self._place(date, self._validate(bets, opps))
        self._settle(self.end_date + pd.Timedelta(days=1))
        history = pd.DataFrame(self.history).set_index("Date")
        return Result(self.games, self.bankroll, history)

    def _settle(self, date: pd.Timestamp) -> pd.DataFrame:
        """Pay out the games played since the last call, and return them as the increment."""
        games = self.games.loc[(self.games["Date"] > self.last_seen) & (self.games["Date"] < date)]
        if not games.empty:
            self.last_seen = games["Date"].max()
            stakes = games[BET_COLS].values
            odds = games[ODDS_COLS].values
            results = games[RESULT_COLS].values
            self.bankroll += (stakes * results * odds).sum(axis=1).sum()
            self._record(date + pd.Timedelta(hours=6), 0.0)
        return games.drop(["Open", *BET_COLS], axis=1)

    def _options(self, date: pd.Timestamp) -> pd.DataFrame:
        opps = self.games.loc[(self.games["Open"] <= date) & (self.games["Date"] >= date)]
        opps = opps.loc[opps[ODDS_COLS].sum(axis=1) > 0]
        return opps.drop([*RESULT_COLS, *FEATURE_COLS, "Open"], axis=1)

    def _validate(self, bets: pd.DataFrame, opps: pd.DataFrame) -> pd.DataFrame:
        rows = bets.index.intersection(opps.index)
        cols = bets.columns.intersection(BET_COLS)
        valid = bets.loc[rows, cols]
        valid[valid < self.min_bet] = 0.0
        valid[valid > self.max_bet] = 0.0
        if valid.sum().sum() > self.bankroll:
            valid.loc[:, :] = 0.0
        return valid

    def _place(self, date: pd.Timestamp, bets: pd.DataFrame) -> None:
        self.games.loc[bets.index, BET_COLS] = self.games.loc[bets.index, BET_COLS].add(
            bets, fill_value=0
        )
        self.bankroll -= bets.values.sum()
        self._record(date + pd.Timedelta(hours=23), bets.values.sum())

    def _summary(self, date: pd.Timestamp) -> pd.DataFrame:
        summary = {
            "Bankroll": self.bankroll,
            "Date": date,
            "Min_bet": self.min_bet,
            "Max_bet": self.max_bet,
        }
        return pd.Series(summary).to_frame().T

    def _record(self, date: pd.Timestamp, cash_invested: float) -> None:
        self.history["Date"].append(date)
        self.history["Bankroll"].append(self.bankroll)
        self.history["Cash_Invested"].append(cash_invested)
