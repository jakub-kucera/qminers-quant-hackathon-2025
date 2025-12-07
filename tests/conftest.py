import os

# One thread, as on the evaluation server; set before numpy and torch load their thread pools.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from qqh.environment import Environment
from qqh.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ORIGINALS = ROOT / "originals"
ORIGINAL_MODULES = {p.stem for p in ORIGINALS.glob("*.py")}

torch.set_num_threads(1)


def read_games(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=["Date", "Open"])


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA / "games.csv").exists():
        pytest.skip("no data: run `uv run python scripts/fetch_data.py`")
    return DATA


@pytest.fixture(scope="session")
def early_seasons(data_dir) -> pd.DataFrame:
    """Five early seasons of the Bookie Rookie data (it has draw odds), about 1,800 games."""
    games = read_games(data_dir / "bookie_rookie_games.csv")
    return games[games["Season"] <= 1993]


def load_original(module: str):
    """A model from originals/, imported fresh: the files import each other as `model`."""
    for name in [m for m in sys.modules if m.split(".")[0] in ORIGINAL_MODULES]:
        del sys.modules[name]
    sys.path.insert(0, str(ORIGINALS))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(ORIGINALS))


# Settings that open the models' betting windows, so that they bet on the early test seasons.
OPEN_WINDOW = {
    "hybrid": {"start_season": None},
    "hybrid-retuned": {"start_season": None},
    "simple-rolling": {"start_season": None, "end_season": None},
}


def run(model, games: pd.DataFrame):
    return Environment(games, model, init_bankroll=1000, min_bet=1, max_bet=250).run()


@pytest.fixture(scope="session")
def package_runs(early_seasons):
    """Each package model's run on the early seasons, computed once per test session."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = run(MODELS[name].build(**OPEN_WINDOW.get(name, {})), early_seasons)
        return cache[name]

    return get
