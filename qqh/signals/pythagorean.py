from collections import defaultdict


class Pythagorean:
    """Pythagorean expectation: GF^p / (GF^p + GA^p) over each team's last 100 games,
    decayed by age and regularised by a constant added to both totals.

    Two features, the home and the away team's expectation.
    """

    def __init__(self, power=1.0, regularization=20, daily_decay=0.992):
        self.power = power
        self.regularization = regularization
        self.daily_decay = daily_decay
        self.games = defaultdict(list)  # team -> [date, scored, allowed]

    def add_game(self, game):
        date = game["Date"]
        self.games[game["HID"]].append([date, game["HS"], game["AS"]])
        self.games[game["AID"]].append([date, game["AS"], game["HS"]])

    def _weighted(self, team_id, column, date):
        return sum(
            [
                x[column] * (self.daily_decay ** abs((date - x[0]).days))
                for x in self.games[team_id][-100:]
            ]
        )

    def _expectation(self, team_id, date):
        scored = self.regularization + self._weighted(team_id, 1, date)
        allowed = self.regularization + self._weighted(team_id, 2, date)
        return (scored**self.power) / (scored**self.power + allowed**self.power)

    def features(self, home_id, away_id, season, date):
        return [self._expectation(home_id, date), self._expectation(away_id, date)]
