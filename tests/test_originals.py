"""The package's models place exactly the bets the team's original files place.

originals/ holds each model as the team left it; qqh/ is the same code cleaned up.
Each case runs both through the same days and compares every stake.
"""

import pandas as pd
import pytest
from conftest import load_original, run

from qqh.environment import BET_COLS

# model name -> (module in originals/, kwargs for the original matching conftest.OPEN_WINDOW)
CASES = {
    "ensemble": ("model", {}),
    "hybrid": ("model_hybrid", {"start_season": None}),
    "bookie-rookie": ("model_bookie_rookie", {}),
    "simple-rolling": ("model_simple_rolling", {"start_season": None, "end_year": None}),
    "underdog-sniper": ("model_underdog_sniper", {}),
    "market-residual": ("marek_model", {}),
    "no-bets": ("model_empty_zeros", {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_bets_as_the_original(name, early_seasons, package_runs):
    module, kwargs = CASES[name]
    expected = run(load_original(module).Model(**kwargs), early_seasons)
    actual = package_runs(name)
    pd.testing.assert_frame_equal(actual.games[BET_COLS], expected.games[BET_COLS])
    assert actual.bankroll == expected.bankroll
    if name != "no-bets":
        assert actual.staked > 0, "the slice should be long enough for the model to bet"
