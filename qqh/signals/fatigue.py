from collections import defaultdict


class Fatigue:
    """Rest and travel: back-to-back games, one day of rest, and road trips of 2+ away games.

    One feature, positive when the away team is the more tired: the home team's penalties
    count against it and the away team's for it.
    """

    def __init__(self, b2b_penalty=1.5, one_day_penalty=0.8, road_trip_penalty=0.5):
        self.b2b_penalty = b2b_penalty
        self.one_day_penalty = one_day_penalty
        self.road_trip_penalty = road_trip_penalty
        self.last_game = {}
        self.recent = defaultdict(list)  # team -> its last 20 games as (date, at home)

    def add_game(self, game):
        date = game["Date"]
        home_id = game["HID"]
        away_id = game["AID"]
        self.last_game[home_id] = date
        self.last_game[away_id] = date
        self.recent[home_id].append({"date": date, "is_home": True})
        self.recent[away_id].append({"date": date, "is_home": False})
        if len(self.recent[home_id]) > 20:
            self.recent[home_id] = self.recent[home_id][-20:]
        if len(self.recent[away_id]) > 20:
            self.recent[away_id] = self.recent[away_id][-20:]

    def _road_trip(self, team_id, date):
        """Penalty for consecutive away games within the last two weeks."""
        if team_id not in self.recent or len(self.recent[team_id]) < 2:
            return 0.0
        consecutive_away = 0
        for game in reversed(self.recent[team_id][-10:]):
            if (date - game["date"]).days <= 14:
                if not game["is_home"]:
                    consecutive_away += 1
                else:
                    break
        if consecutive_away >= 3:
            return -self.road_trip_penalty
        elif consecutive_away >= 2:
            return -self.road_trip_penalty * 0.6
        return 0.0

    def features(self, home_id, away_id, season, date):
        if home_id not in self.last_game or away_id not in self.last_game:
            return None
        home_days = (date - self.last_game[home_id]).days
        away_days = (date - self.last_game[away_id]).days
        factor = 0.0
        if home_days == 0:
            factor += self.b2b_penalty
        elif home_days == 1:
            factor += self.one_day_penalty
        if away_days == 0:
            factor -= self.b2b_penalty
        elif away_days == 1:
            factor -= self.one_day_penalty
        factor += self._road_trip(home_id, date) - self._road_trip(away_id, date)
        return [factor]
