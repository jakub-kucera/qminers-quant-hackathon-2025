"""The team's models, each with the settings it was last run with."""

import sys
from dataclasses import dataclass

from qqh.models.ensemble import EnsembleModel
from qqh.models.hybrid import HybridModel
from qqh.models.market_residual import MarketResidualNet
from qqh.models.no_bets import NoBets
from qqh.models.simple_rolling import SimpleRollingModel
from qqh.models.underdog_sniper import UnderdogSniper


@dataclass(frozen=True)
class Entry:
    cls: type
    preset: str | None  # name of a settings dict in the class's module
    author: str
    summary: str

    @property
    def settings(self) -> dict:
        if self.preset is None:
            return {}
        return dict(getattr(sys.modules[self.cls.__module__], self.preset))

    def build(self, **overrides):
        return self.cls(**{**self.settings, **overrides})


MODELS = {
    "ensemble": Entry(
        EnsembleModel, "QUALIFICATION", "Jakub Kucera",
        "seven signals stacked by multinomial logistic regression, fractional Kelly",
    ),
    "hybrid": Entry(
        HybridModel, "FINALS_ROUND_2", "Jakub Kucera, Přemysl Pilař",
        "the ensemble's probabilities with Pepe's confidence-weighted Kelly",
    ),
    "bookie-rookie": Entry(
        HybridModel, "BOOKIE_ROOKIE", "Jakub Kucera, Přemysl Pilař",
        "the hybrid tuned for the Bookie Rookie task",
    ),
    "simple-rolling": Entry(
        SimpleRollingModel, "FINALS_DAY", "Jakub Kucera",
        "rolling form, Elo and head-to-head in one logistic regression, home bets only",
    ),
    "underdog-sniper": Entry(
        UnderdogSniper, None, "Přemysl Pilař",
        "calibrated probabilities, flat bets on underdogs at odds of 3 or more",
    ),
    "market-residual": Entry(
        MarketResidualNet, None, "Marek Vedral",
        "a network correcting the bookmaker's log-odds, simultaneous Kelly",
    ),
    "no-bets": Entry(NoBets, None, "", "places no bets"),
}  # fmt: skip
