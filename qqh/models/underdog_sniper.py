"""Pepe's underdog sniper, the finals' round-3 submission: flat bets on long-shot underdogs."""

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from qqh.models.ensemble import AWAY, DRAW, HOME, empty_bets
from qqh.signals import Fatigue, HockeyFourFactor, NateSilverElo, PoissonGoals, RollingForm


class UnderdogSniper:
    """Five signals into a multinomial logistic regression, home and away probabilities
    recalibrated by an isotonic regression on the model's own past predictions, and a flat
    stake on every underdog (odds >= `min_odds`) with an expected value of `ev_threshold`.

    Known flaws, kept because this is what was submitted (see docs/postmortem.md):
    - training rows are built after the game is added to the signals, so they see the result;
    - the classifier and the calibrator refit only when their sample counts land exactly on a
      multiple of 100 at the start of a day, which may not happen for a long time.
    """

    def __init__(
        self,
        calibration_window=1000,
        min_calibration_samples=100,
        fixed_bet_size=250.0,
        ev_threshold=0.20,
        min_odds=3.0,
    ):
        self.signals = [
            NateSilverElo(k_base=20, home_advantage=40, margin_cap=5),
            RollingForm(window_short=3, window_med=11, window_long=22),
            PoissonGoals(regularization=3.5, daily_decay=0.995),
            Fatigue(b2b_penalty=1.6, one_day_penalty=0.8),
            HockeyFourFactor(lr_required_n=400, daily_decay=0.99),
        ]
        self.classifier = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        self.is_fitted = False
        self.X_train = []
        self.y_train = []
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.calibration_probs = []
        self.calibration_outcomes = []
        self.calibration_window = calibration_window
        self.min_calibration_samples = min_calibration_samples
        self.is_calibrated = False
        self.fixed_bet_size = fixed_bet_size
        self.ev_threshold = ev_threshold
        self.min_odds = min_odds
        self.predicted = {}  # game ID -> raw probabilities, to calibrate on once it's played

    def _features(self, home_id, away_id, season, date):
        features = []
        for signal in self.signals:
            values = signal.features(home_id, away_id, season, date)
            if values is None:
                return None
            features.extend(values)
        return features

    def _ingest(self, inc: pd.DataFrame):
        for idx, game in inc.iterrows():
            for signal in self.signals:
                signal.add_game(game)
            outcome = HOME if game["H"] else AWAY if game["A"] else DRAW
            features = self._features(game["HID"], game["AID"], game["Season"], game["Date"])
            if features is not None:
                self.X_train.append(features)
                self.y_train.append(outcome)
            if idx in self.predicted:
                probs = self.predicted[idx]
                self.calibration_probs.append(probs[HOME])
                self.calibration_outcomes.append(1 if game["H"] else 0)
                self.calibration_probs.append(probs[AWAY])
                self.calibration_outcomes.append(1 if game["A"] else 0)
                if len(self.calibration_probs) > self.calibration_window * 2:
                    start = len(self.calibration_probs) - self.calibration_window * 2
                    self.calibration_probs = self.calibration_probs[start:]
                    self.calibration_outcomes = self.calibration_outcomes[start:]

    def _refit(self):
        if len(self.y_train) >= 500 and len(self.y_train) % 100 == 0:
            self.classifier.fit(self.X_train, self.y_train)
            self.is_fitted = True
        if len(self.calibration_probs) >= self.min_calibration_samples:
            if len(self.calibration_probs) % 100 == 0:
                self.calibrator.fit(self.calibration_probs, self.calibration_outcomes)
                self.is_calibrated = True

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        self._ingest(inc)
        self._refit()
        bets = empty_bets(opps)
        if not self.is_fitted:
            return bets
        for idx, game in opps.iterrows():
            features = self._features(game["HID"], game["AID"], game["Season"], game["Date"])
            if features is None:
                continue
            try:
                probs = self.classifier.predict_proba([features])[0]
            except Exception:
                continue
            self.predicted[idx] = probs
            p_home, p_away = probs[HOME], probs[AWAY]
            if self.is_calibrated:
                p_away = max(0.001, self.calibrator.transform([p_away])[0])
                p_home = max(0.001, self.calibrator.transform([p_home])[0])
            for column, odds, p in (("BetH", "OddsH", p_home), ("BetA", "OddsA", p_away)):
                if game[odds] >= self.min_odds and (game[odds] * p) - 1.0 >= self.ev_threshold:
                    amount = min(self.fixed_bet_size, max_bet, bankroll)
                    if amount >= min_bet:
                        bets.at[idx, column] = amount
                        bankroll -= amount
        return bets
