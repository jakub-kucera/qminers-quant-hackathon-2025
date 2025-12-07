import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from qqh.signals._stats import inverse_sigmoid, stat

# Box-score columns behind each tracked stat; "{}" is the side, H or A.
TRACKED = {
    "ShotsOnGoal": "{}_SOG",
    "Goals": "{}S",
    "Penalties": "{}_PEN",
    "PowerPlayGoals": "{}_PPG",
    "Saves": "{}_SV",
    "FaceoffsWon": "{}_FO",
    "Hits": "{}_HIT",
    "Blocks": "{}_BLK",
    "BlockedShots": "{}_BLK_S",
}


def _side(game, side):
    entry = {"Date": game["Date"]}
    for name, column in TRACKED.items():
        entry[name] = stat(game, column.format(side))
    return entry


def _has_bad(values):
    return any(math.isnan(v) or math.isinf(v) for v in values)


class HockeyFourFactor:
    """Dean Oliver's "four factors", rebuilt from hockey box scores.

    Per team, time-decayed sums of its own and its opponents' stats give nine rates
    (shooting and save percentage, power play and penalty kill, faceoffs, a Corsi-like
    share, shot differential, goals for and against). A logistic regression, refitted every
    200 games, maps one team's rates to its chance of winning; the feature is the
    difference of the two teams' log-odds.
    """

    def __init__(self, lr_required_n=500, daily_decay=0.994):
        self.lr_required_n = lr_required_n
        self.daily_decay = daily_decay
        self.team_stats = defaultdict(list)
        self.opponent_stats = defaultdict(list)
        self.past_pred = []
        self.lr = None
        self.lr_retrain = 0

    def _totals(self, date, stats):
        totals = dict.fromkeys(TRACKED, 0)
        totals["Weight"] = 0
        for entry in stats:
            weight = self.daily_decay ** abs((date - entry["Date"]).days)
            for name in TRACKED:
                totals[name] += entry[name] * weight
            totals["Weight"] += weight
        return totals

    def add_game(self, game):
        home = self._team_rates(game["HID"], game["Date"])
        away = self._team_rates(game["AID"], game["Date"])
        if home is not None and not _has_bad(home):
            self.past_pred.append([*home, game["H"]])
            self.lr_retrain -= 1
        if away is not None and not _has_bad(away):
            self.past_pred.append([*away, game["A"]])
            self.lr_retrain -= 1
        self.team_stats[game["HID"]].append(_side(game, "H"))
        self.team_stats[game["AID"]].append(_side(game, "A"))
        self.opponent_stats[game["AID"]].append(_side(game, "H"))
        self.opponent_stats[game["HID"]].append(_side(game, "A"))

    def _team_rates(self, team_id, date):
        if len(self.team_stats[team_id]) <= 5:
            return None
        own = self._totals(date, self.team_stats[team_id][-100:])
        opp = self._totals(date, self.opponent_stats[team_id][-100:])
        if own["Weight"] < 0.01 or opp["Weight"] < 0.01:
            return None
        corsi_for = own["ShotsOnGoal"] + own["BlockedShots"]
        corsi_against = opp["ShotsOnGoal"] + opp["BlockedShots"]
        return [
            own["Goals"] / max(own["ShotsOnGoal"], 0.1),  # shooting %
            own["Saves"] / max(opp["ShotsOnGoal"], 0.1),  # save %
            own["PowerPlayGoals"] / max(opp["Penalties"], 0.1),  # power play
            1.0 - (opp["PowerPlayGoals"] / max(own["Penalties"], 0.1)),  # penalty kill
            own["FaceoffsWon"] / max(own["FaceoffsWon"] + opp["FaceoffsWon"], 0.1),
            corsi_for / max(corsi_for + corsi_against, 0.1),
            (own["ShotsOnGoal"] - opp["ShotsOnGoal"]) / own["Weight"],  # shot differential
            own["Goals"] / own["Weight"],
            opp["Goals"] / opp["Weight"],
        ]

    def features(self, home_id, away_id, season, date):
        home = self._team_rates(home_id, date)
        away = self._team_rates(away_id, date)
        if home is None or away is None:
            return None
        if _has_bad(home + away):
            return None
        if len(self.past_pred) < self.lr_required_n:
            return None
        if self.lr_retrain <= 0:
            self.lr_retrain = 200
            valid = [entry for entry in self.past_pred if not _has_bad(entry)]
            if len(valid) < self.lr_required_n:
                return None
            data = np.array(valid)
            weights = np.exp(-0.0003 * np.arange(len(valid)))
            self.lr = LogisticRegression(max_iter=10000)
            self.lr.fit(data[:, :-1], data[:, -1], sample_weight=weights[::-1])
        return [
            inverse_sigmoid(self.lr.predict_proba(np.array([home]))[0, 1])
            - inverse_sigmoid(self.lr.predict_proba(np.array([away]))[0, 1])
        ]
