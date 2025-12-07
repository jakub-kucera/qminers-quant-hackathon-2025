"""Helpers shared by the signals."""

import math

import pandas as pd


def stat(game, column):
    """A box-score value, 0 when the column is missing or empty (older seasons lack some)."""
    value = game.get(column, 0)
    return 0 if pd.isna(value) else value


def inverse_sigmoid(x):
    """Log-odds of a probability, clamped away from 0 and 1."""
    x = max(0.001, min(0.999, x))
    return math.log(x / (1 - x))
