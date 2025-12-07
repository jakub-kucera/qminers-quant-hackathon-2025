"""tools/bundle.py makes a self-contained file that bets exactly like the package."""

import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import OPEN_WINDOW, ROOT, run

from qqh.environment import BET_COLS
from qqh.models import MODELS

sys.path.insert(0, str(ROOT / "tools"))
from bundle import bundle  # noqa: E402

# What the evaluation server had installed, plus the standard library.
SERVER_PACKAGES = {"numpy", "pandas", "sklearn", "scipy", "torch", "statsmodels", "xgboost"}


def imported_modules(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    return names


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"bundled_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", list(MODELS))
def test_bundle_bets_like_the_package(name, tmp_path, early_seasons, package_runs):
    path = tmp_path / f"{name.replace('-', '_')}.py"
    path.write_text(bundle(name))
    outside = imported_modules(path.read_text()) - set(sys.stdlib_module_names)
    assert outside <= SERVER_PACKAGES

    module = load(path)
    module.Model()  # what the server does
    entry = MODELS[name]
    cls = getattr(module, entry.cls.__name__)
    actual = run(cls(**{**entry.settings, **OPEN_WINDOW.get(name, {})}), early_seasons)
    expected = package_runs(name)
    pd.testing.assert_frame_equal(actual.games[BET_COLS], expected.games[BET_COLS])
