from collections import defaultdict

import numpy as np

from qqh.signals._stats import stat


class RollingForm:
    """Form over the last few games, in three windows (short, medium, long).

    Per window, seven home-minus-away differences: goals for, goals against, shots on goal,
    power-play %, faceoff %, win % and shooting %.
    """

    def __init__(self, window_short=5, window_med=10, window_long=20):
        self.windows = [window_short, window_med, window_long]
        self.window_long = window_long
        self.history = defaultdict(list)

    def add_game(self, game):
        faceoffs = stat(game, "H_FO") + stat(game, "A_FO")
        home = {
            "goals_for": game["HS"],
            "goals_against": game["AS"],
            "sog_for": stat(game, "H_SOG"),
            "sog_against": stat(game, "A_SOG"),
            "ppg": stat(game, "H_PPG"),
            "pp_opp": stat(game, "A_PEN"),
            "faceoff_wins": stat(game, "H_FO"),
            "faceoff_total": faceoffs,
            "win": 1 if game["H"] else 0,
        }
        away = {
            "goals_for": game["AS"],
            "goals_against": game["HS"],
            "sog_for": stat(game, "A_SOG"),
            "sog_against": stat(game, "H_SOG"),
            "ppg": stat(game, "A_PPG"),
            "pp_opp": stat(game, "H_PEN"),
            "faceoff_wins": stat(game, "A_FO"),
            "faceoff_total": faceoffs,
            "win": 1 if game["A"] else 0,
        }
        keep = int(self.window_long * 1.5)
        for team_id, entry in ((game["HID"], home), (game["AID"], away)):
            self.history[team_id].append(entry)
            if len(self.history[team_id]) > keep:
                self.history[team_id] = self.history[team_id][-keep:]

    def _window(self, team_id, size):
        if team_id not in self.history or len(self.history[team_id]) < size:
            return None
        games = self.history[team_id][-size:]
        return [
            np.mean([g["goals_for"] for g in games]),
            np.mean([g["goals_against"] for g in games]),
            np.mean([g["sog_for"] for g in games]),
            np.mean([g["sog_against"] for g in games]),
            sum(g["ppg"] for g in games) / max(sum(g["pp_opp"] for g in games), 0.1),
            sum(g["faceoff_wins"] for g in games)
            / max(sum(g["faceoff_total"] for g in games), 0.1),
            np.mean([g["win"] for g in games]),
            sum(g["goals_for"] for g in games) / max(sum(g["sog_for"] for g in games), 0.1),
        ]

    def features(self, home_id, away_id, season, date):
        features = []
        for size in self.windows:
            home = self._window(home_id, size)
            away = self._window(away_id, size)
            if home is None or away is None:
                return None
            # sog_against (index 3) is computed but, as in the submitted model, not used
            features.extend(home[i] - away[i] for i in (0, 1, 2, 4, 5, 6, 7))
        return features
