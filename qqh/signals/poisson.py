from collections import defaultdict

LEAGUE_AVERAGE_GOALS = 3.0


class PoissonGoals:
    """Expected goals from decayed scoring and conceding rates over each team's last 50 games:
    a team's attack rate, relative to the league average, times the opponent's defence rate.

    The feature is the expected goal difference.
    """

    def __init__(self, regularization=3.0, daily_decay=0.992):
        self.regularization = regularization  # unused, kept so tuned parameter sets still load
        self.daily_decay = daily_decay
        self.scored = defaultdict(list)  # team -> [date, goals]
        self.allowed = defaultdict(list)

    def add_game(self, game):
        date = game["Date"]
        home_id = game["HID"]
        away_id = game["AID"]
        self.scored[home_id].append([date, game["HS"]])
        self.scored[away_id].append([date, game["AS"]])
        self.allowed[home_id].append([date, game["AS"]])
        self.allowed[away_id].append([date, game["HS"]])
        if len(self.scored[home_id]) > 100:
            self.scored[home_id] = self.scored[home_id][-100:]
            self.allowed[home_id] = self.allowed[home_id][-100:]
        if len(self.scored[away_id]) > 100:
            self.scored[away_id] = self.scored[away_id][-100:]
            self.allowed[away_id] = self.allowed[away_id][-100:]

    def _rate(self, team_id, date, offence):
        if team_id not in self.scored:
            return None
        data = self.scored[team_id] if offence else self.allowed[team_id]
        if len(data) < 5:
            return None
        goals = 0.0
        total_weight = 0.0
        for game_date, count in data[-50:]:
            weight = self.daily_decay ** abs((date - game_date).days)
            goals += count * weight
            total_weight += weight
        if total_weight < 0.01:
            return None
        return goals / total_weight

    def features(self, home_id, away_id, season, date):
        home_attack = self._rate(home_id, date, offence=True)
        away_defence = self._rate(away_id, date, offence=False)
        away_attack = self._rate(away_id, date, offence=True)
        home_defence = self._rate(home_id, date, offence=False)
        if any(x is None for x in [home_attack, away_defence, away_attack, home_defence]):
            return None
        expected_home = (home_attack / LEAGUE_AVERAGE_GOALS) * away_defence
        expected_away = (away_attack / LEAGUE_AVERAGE_GOALS) * home_defence
        return [expected_home - expected_away]
