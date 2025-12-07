"""A deliberately small model: rolling form, Elo and the last head-to-head into one binary
logistic regression (home win or not), betting fractional Kelly on the home team only.

Built on the finals day as a fast alternative to the ensemble; never submitted.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Tuned on the finals day on rounds 1 and 2 (seasons 2015-2018, in sample).
FINALS_DAY = dict(
    start_season=2015,
    end_season=2018,
    min_edge=0.0127,
    max_kelly_fraction=0.4391,
    kelly_multiplier=1.8701,
    normalize_odds=False,
    min_probability=0.7002,
)


def _value(game, column, default=0):
    try:
        value = game.get(column, default)
        return default if pd.isna(value) else value
    except Exception:
        return default


class FormEloH2H:
    """Per-team form over three windows, an Elo with overtime results worth 0.75/0.25, and
    who won the teams' last meeting. Missing box-score values get rough defaults."""

    def __init__(self, window_short=5, window_med=10, window_long=20, elo_k=20,
                 elo_home_advantage=50):  # fmt: skip
        self.windows = [window_short, window_med, window_long]
        self.history = defaultdict(list)
        self.elo = defaultdict(lambda: 1500)
        self.elo_k = elo_k
        self.elo_home_advantage = elo_home_advantage
        self.last_meeting = {}  # (home, away) -> 1.0 if home won their last game

    def add_game(self, game):
        try:
            home_id, away_id = game["HID"], game["AID"]
            home_score, away_score = game["HS"], game["AS"]
            if pd.isna(home_id) or pd.isna(away_id) or pd.isna(home_score) or pd.isna(away_score):
                return
        except KeyError:
            return
        faceoffs = max(_value(game, "H_FO", 30) + _value(game, "A_FO", 30), 1)
        self.history[home_id].append({
            "goals_for": home_score,
            "goals_against": away_score,
            "sog_for": _value(game, "H_SOG", home_score * 3),
            "sog_against": _value(game, "A_SOG", away_score * 3),
            "pp_goals": _value(game, "H_PPG", 0),
            "pp_opps": max(_value(game, "H_PEN", 3), 1),
            "faceoffs_won": _value(game, "H_FO", 30),
            "faceoffs_total": faceoffs,
            "won": 1 if _value(game, "H", False) else 0,
        })  # fmt: skip
        self.history[away_id].append({
            "goals_for": away_score,
            "goals_against": home_score,
            "sog_for": _value(game, "A_SOG", away_score * 3),
            "sog_against": _value(game, "H_SOG", home_score * 3),
            "pp_goals": _value(game, "A_PPG", 0),
            "pp_opps": max(_value(game, "A_PEN", 3), 1),
            "faceoffs_won": _value(game, "A_FO", 30),
            "faceoffs_total": faceoffs,
            "won": 1 if _value(game, "A", False) else 0,
        })  # fmt: skip
        self._update_elo(home_id, away_id, home_score, away_score, game)
        home_won = 1.0 if home_score > away_score else 0.0
        self.last_meeting[(home_id, away_id)] = home_won
        self.last_meeting[(away_id, home_id)] = 1.0 - home_won

    def _update_elo(self, home_id, away_id, home_score, away_score, game):
        home_elo = self.elo[home_id]
        away_elo = self.elo[away_id]
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + self.elo_home_advantage)) / 400))
        expected_away = 1 - expected_home
        overtime = game.get("Special", "") in ["OT", "SO"]
        if home_score > away_score:
            actual_home, actual_away = (0.75, 0.25) if overtime else (1.0, 0.0)
        else:
            actual_home, actual_away = (0.25, 0.75) if overtime else (0.0, 1.0)
        self.elo[home_id] += self.elo_k * (actual_home - expected_home)
        self.elo[away_id] += self.elo_k * (actual_away - expected_away)

    def _window(self, team_id, size):
        if team_id not in self.history:
            return None
        games = self.history[team_id][-size:]
        if len(games) < min(3, size):
            return None
        n = len(games)
        goals_for = sum(g["goals_for"] for g in games)
        sog_for = sum(g["sog_for"] for g in games)
        return [
            goals_for / n,
            sum(g["goals_against"] for g in games) / n,
            sog_for / n,
            sum(g["sog_against"] for g in games) / n,
            sum(g["pp_goals"] for g in games) / max(sum(g["pp_opps"] for g in games), 0.1),
            sum(g["faceoffs_won"] for g in games) / max(sum(g["faceoffs_total"] for g in games), 1),
            sum(g["won"] for g in games) / n,
            goals_for / max(sog_for, 0.1),
        ]

    def features(self, home_id, away_id):
        features = []
        for size in self.windows:
            home = self._window(home_id, size)
            away = self._window(away_id, size)
            if home is None or away is None:
                return None
            features.extend(home[i] - away[i] for i in (0, 1, 2, 4, 5, 6, 7))
        elo_gap = (self.elo.get(home_id, 1500) + self.elo_home_advantage) - self.elo.get(
            away_id, 1500
        )
        features.append(elo_gap / 100)
        features.append(self.last_meeting.get((home_id, away_id), 0.5))
        return features


class SimpleRollingModel:
    def __init__(
        self,
        rolling_window_short=5,
        rolling_window_med=10,
        rolling_window_long=20,
        required_games=100,
        retrain_freq=50,
        start_season=None,
        end_season=None,
        min_edge=0.02,
        max_kelly_fraction=0.25,
        kelly_multiplier=1.0,
        normalize_odds=False,
        min_probability=0.5,
    ):
        """Only seasons from `start_season` to `end_season` are learned from and bet on."""
        self.required_games = required_games
        self.retrain_freq = retrain_freq
        self.retrain_in = retrain_freq
        self.start_season = start_season
        self.end_season = end_season
        self.min_edge = min_edge
        self.max_kelly_fraction = max_kelly_fraction
        self.kelly_multiplier = kelly_multiplier
        self.normalize_odds = normalize_odds
        self.min_probability = min_probability
        self.signal = FormEloH2H(rolling_window_short, rolling_window_med, rolling_window_long)
        self.training_data = []
        self.model = None
        self.scaler = StandardScaler()

    def _in_range(self, season):
        if self.start_season and season < self.start_season:
            return False
        if self.end_season and season > self.end_season:
            return False
        return True

    def _add_game(self, game):
        if not self._in_range(game["Season"]):
            return
        features = self.signal.features(game["HID"], game["AID"])
        if features is not None:
            self.training_data.append(features + [1 if game["H"] else 0])
            self.retrain_in -= 1
        self.signal.add_game(game)

    def _train(self):
        if len(self.training_data) < self.required_games:
            return
        data = np.array(self.training_data)
        x = self.scaler.fit_transform(data[:, :-1])
        weights = np.exp(-0.0003 * np.arange(len(data)))[::-1]
        self.model = LogisticRegression(max_iter=10000, solver="lbfgs", random_state=42)
        self.model.fit(x, data[:, -1], sample_weight=weights)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        for idx in inc.index:
            self._add_game(inc.loc[idx])
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        bets = pd.DataFrame(
            data=np.zeros((len(opps), 3)), columns=["BetH", "BetA", "BetD"], index=opps.index
        )
        if len(self.training_data) < self.required_games:
            return bets
        if self.retrain_in <= 0:
            self._train()
            self.retrain_in = self.retrain_freq
        for i in opps.index:
            game = opps.loc[i]
            if not self._in_range(int(game["Season"])):
                continue
            features = self.signal.features(game["HID"], game["AID"])
            if features is None or self.model is None:
                continue
            prob_home = self.model.predict_proba(self.scaler.transform([features]))[0][1]
            odds_home = game["OddsH"]
            if self.normalize_odds:
                odds_draw = game.get("OddsD", 0)
                total = 1.0 / odds_home + 1.0 / game["OddsA"]
                total += 1.0 / odds_draw if odds_draw > 0 else 0
                implied = (1.0 / odds_home) / total
            else:
                implied = 1.0 / odds_home
            edge = prob_home - implied
            if edge > self.min_edge and prob_home > self.min_probability:
                kelly = edge / (odds_home - 1) * self.kelly_multiplier
                kelly = max(0, min(kelly, self.max_kelly_fraction))
                amount = max(min_bet, min(kelly * bankroll, max_bet))
                bets.at[i, "BetH"] = np.round(amount)
        return bets
