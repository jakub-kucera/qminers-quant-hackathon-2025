"""The finals' round-1 submission: no bets.

The ensemble's first day replays the whole history and, on the evaluation server's single
CPU, wasn't going to finish in time, so this went in instead.
"""

import numpy as np
import pandas as pd


class NoBets:
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        return pd.DataFrame(
            data=np.zeros((len(opps), 3)), columns=["BetH", "BetD", "BetA"], index=opps.index
        )
