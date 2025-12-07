"""The competition's evaluation rounds, rebuilt from the organisers' published data.

Finals rounds are exact: each round's file is what the server replayed (the whole history with
its odds set to zero, then the round's two seasons with odds). The qualification round is an
approximation: its hidden seasons (2011-2014) were published later inside the finals' training
data, whose odds for earlier seasons differ from the qualification data's.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Round:
    name: str
    csv: str
    seasons: tuple[int, int]  # the seasons bet on
    bankroll: float
    min_bet: float
    max_bet: float
    exact: bool

    def games(self) -> pd.DataFrame:
        return pd.read_csv(DATA / self.csv, index_col=0, parse_dates=["Date", "Open"])

    def start(self, games: pd.DataFrame) -> pd.Timestamp | None:
        """First day of the round; the history before it arrives as the first increment."""
        if self.exact:
            return None  # the file's own first opening date
        return games.loc[games["Season"] == self.seasons[0], "Open"].min()


ROUNDS = {
    "qualification": Round(
        "qualification", "finals/train_games.csv", (2011, 2014), 1000, 5, 100, exact=False
    ),
    "finals-1": Round("finals-1", "finals/round1_games.csv", (2015, 2016), 1000, 1, 250, True),
    "finals-2": Round("finals-2", "finals/round2_games.csv", (2017, 2018), 1000, 1, 250, True),
    "finals-3": Round("finals-3", "finals/round3_games.csv", (2019, 2020), 1000, 1, 250, True),
}

# What the team uploaded in each finals round (model name in qqh.models.MODELS), and what the
# evaluation server reported: no bets in any round.
SUBMITTED = {"finals-1": "no-bets", "finals-2": "hybrid", "finals-3": "underdog-sniper"}
ON_THE_SERVER = {"finals-1": "no bets", "finals-2": "no bets", "finals-3": "no bets"}
