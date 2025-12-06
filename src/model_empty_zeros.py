import numpy as np
import pandas as pd


class Model:

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        bets = np.zeros((len(opps), 3))
        bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetD", "BetA"], index=opps.index
        )
        return bets