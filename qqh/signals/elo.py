import math
from collections import defaultdict


class NateSilverElo:
    """FiveThirtyEight-style Elo: a K that grows with the (capped) margin of victory and
    shrinks for a favourite's win, smaller updates for overtime and shootout results, and
    ratings pulled toward the mean between seasons.

    The feature is the rating gap including home advantage.
    """

    def __init__(
        self, k_base=15, home_advantage=35, ot_multiplier=0.7, margin_cap=3, season_carryover=0.75
    ):
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.ot_multiplier = ot_multiplier
        self.margin_cap = margin_cap
        self.season_carryover = season_carryover
        self.ratings = defaultdict(float)
        self.last_season = -1

    def _new_season(self):
        for team in self.ratings:
            self.ratings[team] *= self.season_carryover

    def _win_probability(self, x):
        return 1 / (1 + (math.exp(-x / 175)))

    def _k(self, margin, elo_home, elo_away):
        capped = max(-self.margin_cap, min(self.margin_cap, margin))
        if capped > 0:
            return self.k_base * (abs(capped) + 3) ** 0.8 / (7.5 + 0.006 * (elo_home - elo_away))
        return self.k_base * (abs(capped) + 3) ** 0.8 / (7.5 + 0.006 * (elo_away - elo_home))

    def add_game(self, game):
        home_id = game["HID"]
        away_id = game["AID"]
        if game["Season"] > self.last_season:
            self.last_season = game["Season"]
            self._new_season()
        home_expected = self._win_probability(
            self.ratings[home_id] + self.home_advantage - self.ratings[away_id]
        )
        away_expected = 1 - home_expected
        k = self._k(
            game["HS"] - game["AS"],
            self.ratings[home_id] + self.home_advantage,
            self.ratings[away_id],
        )
        overtime = self.ot_multiplier if game.get("Special", "") in ["OT", "PS", "SO"] else 1.0
        self.ratings[home_id] += k * overtime * (game["H"] - home_expected)
        self.ratings[away_id] += k * overtime * (game["A"] - away_expected)

    def features(self, home_id, away_id, season, date):
        if season > self.last_season:
            self.last_season = season
            self._new_season()
        return [self.ratings[home_id] - self.ratings[away_id] + self.home_advantage]
