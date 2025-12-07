"""qqh.environment replays a season exactly as the organisers' evaluation loop does."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from conftest import read_games

from qqh.environment import BET_COLS, Environment


class Scattershot:
    """Deterministic bets that exercise every rule: valid stakes, stakes under the minimum
    and over the maximum, bets on games that aren't offered, and days that overspend."""

    def __init__(self):
        self.day = 0

    def place_bets(self, summary, opps, inc):
        self.day += 1
        bankroll = summary.iloc[0]["Bankroll"]
        bets = pd.DataFrame(0.0, index=opps.index, columns=["BetH", "BetA", "BetD"])
        for n, i in enumerate(opps.index):
            column = ["BetH", "BetA", "BetD"][(i + self.day) % 3]
            bets.at[i, column] = [5, 0.5, 30, 400, 12][(n + self.day) % 5]
        if self.day % 17 == 0:
            bets.loc[:, "BetH"] = bankroll  # more than the bankroll: the day is rejected
        bets.loc[-1] = [10.0, 0.0, 0.0]  # not on offer: ignored
        return bets


def load_reference(path):
    spec = importlib.util.spec_from_file_location("organisers_environment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Environment


@pytest.mark.parametrize("csv", ["games.csv", "finals/round1_games.csv"])
def test_same_results_as_the_organisers_loop(csv, data_dir, capsys):
    games = read_games(data_dir / csv)
    if csv == "games.csv":
        games = games[games["Season"] <= 1995]
    reference = load_reference(data_dir / "reference" / "environment.py")
    expected_env = reference(
        games.copy(), Scattershot(), init_bankroll=1000, min_bet=1, max_bet=250
    )
    expected_games = expected_env.run()
    actual = Environment(games, Scattershot(), init_bankroll=1000, min_bet=1, max_bet=250).run()
    capsys.readouterr()

    pd.testing.assert_frame_equal(actual.games[BET_COLS], expected_games[BET_COLS])
    assert actual.bankroll == expected_env.bankroll
    pd.testing.assert_frame_equal(actual.history, expected_env.get_history())
    assert actual.staked > 0


def test_the_input_is_not_modified(data_dir):
    games = read_games(data_dir / "games.csv")
    games = games[games["Season"] <= 1991]
    before = games.copy()
    Environment(games, Scattershot(), min_bet=1, max_bet=250).run()
    pd.testing.assert_frame_equal(games, before)
    assert np.isfinite(before["OddsH"]).all()
