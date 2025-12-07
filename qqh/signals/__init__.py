"""Signals: per-team state updated game by game, turned into features for an upcoming game.

Every signal has the same two methods:

- `add_game(game)` takes a finished game (a row of the games table) and updates the state.
- `features(home_id, away_id, season, date)` returns a list of floats for an upcoming game,
  or None while the signal doesn't have enough history yet.

The ensemble, the Pythagorean, margin-rating, four-factor, fatigue and Elo signals are
adapted from the 2024 winners' solution (github.com/AdamBobocky/qminers-24), which predicted
basketball; the hockey adaptations, the Poisson and the rolling-window signals are ours.
"""

from qqh.signals.elo import NateSilverElo
from qqh.signals.fatigue import Fatigue
from qqh.signals.four_factor import HockeyFourFactor
from qqh.signals.margin_ratings import MarginRatings
from qqh.signals.poisson import PoissonGoals
from qqh.signals.pythagorean import Pythagorean
from qqh.signals.rolling import RollingForm

__all__ = [
    "Fatigue",
    "HockeyFourFactor",
    "MarginRatings",
    "NateSilverElo",
    "PoissonGoals",
    "Pythagorean",
    "RollingForm",
]
