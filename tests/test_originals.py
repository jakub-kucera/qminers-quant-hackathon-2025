"""The package's models place exactly the bets the team's original files place.

originals/ holds each model as the team left it; qqh/ is the same code cleaned up.
Each case runs both through the same days and compares every stake.
"""

import pandas as pd
import pytest
from conftest import load_original

from qqh.environment import BET_COLS, Environment
from qqh.models import MODELS

# model name -> (module in originals/, kwargs for the original, overrides for the package)
CASES = {
    "ensemble": ("model", {}, {}),
    "hybrid": ("model_hybrid", {"start_season": None}, {"start_season": None}),
    "bookie-rookie": ("model_bookie_rookie", {}, {}),
    "simple-rolling": (
        "model_simple_rolling",
        {"start_season": None, "end_year": None},
        {"start_season": None, "end_season": None},
    ),
    "underdog-sniper": ("model_underdog_sniper", {}, {}),
    "market-residual": ("marek_model", {}, {}),
    "no-bets": ("model_empty_zeros", {}, {}),
}


def run(model, games: pd.DataFrame):
    env = Environment(games, model, init_bankroll=1000, min_bet=1, max_bet=250)
    return env.run()


@pytest.mark.parametrize("name", list(CASES))
def test_same_bets_as_the_original(name, early_seasons):
    original, original_kwargs, overrides = CASES[name]
    expected = run(load_original(original).Model(**original_kwargs), early_seasons)
    actual = run(MODELS[name].build(**overrides), early_seasons)
    pd.testing.assert_frame_equal(actual.games[BET_COLS], expected.games[BET_COLS])
    assert actual.bankroll == expected.bankroll
    if name != "no-bets":
        assert actual.staked > 0, "the slice should be long enough for the model to bet"
