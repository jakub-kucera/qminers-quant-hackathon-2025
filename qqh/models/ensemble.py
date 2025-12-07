"""The stacked ensemble: seven signals combined by a multinomial logistic regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from qqh.signals import (
    Fatigue,
    HockeyFourFactor,
    MarginRatings,
    NateSilverElo,
    PoissonGoals,
    Pythagorean,
    RollingForm,
)

# Outcome classes, in the order predict_proba returns them.
AWAY, DRAW, HOME = 0, 1, 2

# The qualification round's settings (Optuna on the training seasons, in sample).
QUALIFICATION = dict(
    ensemble_required_n=700,
    ensemble_retrain_freq=300,
    pythagorean_power=0.9815160790897732,
    pythagorean_regularization=10,
    pythagorean_decay=0.992,
    four_factor_required_n=400,
    four_factor_decay=0.9901321052235277,
    gradient_descent_epochs=12,
    gradient_descent_min_games=800,
    exhaustion_b2b_penalty=1.7896362263470917,
    exhaustion_road_penalty=0.6151810640568766,
    elo_k_base=20,
    elo_home_advantage=45,
    elo_margin_cap=5,
    poisson_regularization=3.319620279992762,
    poisson_decay=0.992,
    rolling_window_short=5,
    rolling_window_med=10,
    rolling_window_long=25,
    bet_threshold=0.0419644189979128,
    bet_size_base=0.009610412717873114,
    bet_kelly_fraction=0.10021644282504837,
    bet_min_confidence=0.4485261426475807,
    bet_max_exposure=0.11155183772100152,
    bet_size_max_multiplier=3.042918661869248,
    use_overround_adjustment=False,
    confidence_scaling=True,
)


def empty_bets(opps: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        data=np.zeros((len(opps), 3)), columns=["BetH", "BetA", "BetD"], index=opps.index
    )


class EnsembleModel:
    """Seven signals stacked by a multinomial logistic regression over away/draw/home.

    The stacker is refitted every `ensemble_retrain_freq` games on every game seen so far,
    newer games weighted more (exp(-0.0003 x age in games)). Each game's training row is
    built before the game is added to the signals, so there is no look-ahead.

    Bets: fractional Kelly on each outcome whose probability and edge over the bookmaker's
    implied probability clear their thresholds, scaled by the prediction's certainty and
    capped per game.
    """

    def __init__(
        self,
        ensemble_required_n=700,
        ensemble_retrain_freq=300,
        pythagorean_power=1.0,
        pythagorean_regularization=20,
        pythagorean_decay=0.992,
        four_factor_required_n=500,
        four_factor_decay=0.994,
        gradient_descent_epochs=12,
        gradient_descent_min_games=500,
        exhaustion_b2b_penalty=1.5,
        exhaustion_road_penalty=0.5,
        elo_k_base=15,
        elo_home_advantage=35,
        elo_margin_cap=3,
        poisson_regularization=3.0,
        poisson_decay=0.992,
        rolling_window_short=5,
        rolling_window_med=10,
        rolling_window_long=20,
        bet_threshold=0.05,
        bet_size_base=0.03,
        bet_kelly_fraction=0.25,
        bet_min_confidence=0.4,
        bet_max_exposure=0.15,
        bet_size_max_multiplier=3.0,
        use_overround_adjustment=False,
        confidence_scaling=True,
    ):
        self.signals = [
            Pythagorean(
                power=pythagorean_power,
                regularization=pythagorean_regularization,
                daily_decay=pythagorean_decay,
            ),
            HockeyFourFactor(lr_required_n=four_factor_required_n, daily_decay=four_factor_decay),
            MarginRatings(max_epochs=gradient_descent_epochs, min_games=gradient_descent_min_games),
            Fatigue(b2b_penalty=exhaustion_b2b_penalty, road_trip_penalty=exhaustion_road_penalty),
            NateSilverElo(
                k_base=elo_k_base, home_advantage=elo_home_advantage, margin_cap=elo_margin_cap
            ),
            PoissonGoals(regularization=poisson_regularization, daily_decay=poisson_decay),
            RollingForm(
                window_short=rolling_window_short,
                window_med=rolling_window_med,
                window_long=rolling_window_long,
            ),
        ]
        self.ensemble_required_n = ensemble_required_n
        self.ensemble_retrain_freq = ensemble_retrain_freq
        self.past_pred = []
        self.stacker = None
        self.retrain_in = 0
        self.bet_threshold = bet_threshold
        self.bet_size_base = bet_size_base
        self.bet_kelly_fraction = bet_kelly_fraction
        self.bet_min_confidence = bet_min_confidence
        self.bet_max_exposure = bet_max_exposure
        self.bet_size_max_multiplier = bet_size_max_multiplier
        self.use_overround_adjustment = use_overround_adjustment
        self.confidence_scaling = confidence_scaling

    def _features(self, home_id, away_id, season, date):
        """Every signal's features, or None as soon as one of them has too little history.

        Signals after the first None are not asked, which matters: asking has side effects
        (the margin ratings refit, Elo starts a new season).
        """
        features = []
        for signal in self.signals:
            values = signal.features(home_id, away_id, season, date)
            if values is None:
                return None
            features.extend(values)
        return features

    def _add_game(self, game):
        outcome = HOME if game["H"] else AWAY if game["A"] else DRAW
        features = self._features(game["HID"], game["AID"], game["Season"], game["Date"])
        if features is not None:
            self.past_pred.append([*features, outcome])
            self.retrain_in -= 1
        for signal in self.signals:
            signal.add_game(game)

    def _ingest(self, inc: pd.DataFrame):
        for idx in inc.index:
            self._add_game(inc.loc[idx])

    def _predict(self, game):
        """[P(away), P(draw), P(home)] for an upcoming game, or None if it can't predict yet."""
        if len(self.past_pred) < self.ensemble_required_n:
            return None
        features = self._features(game["HID"], game["AID"], game["Season"], game["Date"])
        if features is None:
            return None
        if self.retrain_in <= 0:
            self.retrain_in = self.ensemble_retrain_freq
            data = np.array(self.past_pred)
            weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
            self.stacker = LogisticRegression(max_iter=10000, solver="lbfgs")
            self.stacker.fit(data[:, :-1], data[:, -1], sample_weight=weights[::-1])
        return self.stacker.predict_proba(np.array([features]))[0]

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        self._ingest(inc)
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        bets = empty_bets(opps)
        for i in opps.index:
            game = opps.loc[i]
            probs = self._predict(game)
            if probs is None:
                continue
            odds = {"BetH": game["OddsH"], "BetA": game["OddsA"], "BetD": game["OddsD"]}
            if any(o <= 1.0 for o in odds.values()):
                continue
            for column, amount in self._stakes(probs, odds, bankroll, min_bet, max_bet):
                bets.at[i, column] = amount
        return bets

    def _stakes(self, probs, odds, bankroll, min_bet, max_bet):
        prob = {"BetH": probs[HOME], "BetA": probs[AWAY], "BetD": probs[DRAW]}
        implied = {column: 1 / o for column, o in odds.items()}
        if self.use_overround_adjustment:
            total = implied["BetH"] + implied["BetA"] + implied["BetD"]
            implied = {column: p / total for column, p in implied.items()}
        epsilon = 1e-10
        entropy = -(
            prob["BetH"] * np.log(prob["BetH"] + epsilon)
            + prob["BetA"] * np.log(prob["BetA"] + epsilon)
            + prob["BetD"] * np.log(prob["BetD"] + epsilon)
        )
        confidence = 1.0 - (entropy / np.log(3))
        multiplier = confidence if self.confidence_scaling else 1.0

        candidates = []
        for column in ("BetH", "BetA", "BetD"):
            p, o = prob[column], odds[column]
            if p < self.bet_min_confidence:
                continue
            edge = p - implied[column]
            if edge <= self.bet_threshold:
                continue
            kelly = max(0, (p * (o - 1) - (1 - p)) / (o - 1))
            fraction = self.bet_kelly_fraction * kelly * multiplier
            fraction = min(fraction, self.bet_size_base * self.bet_size_max_multiplier)
            candidates.append((column, fraction, edge))
        candidates.sort(key=lambda c: c[2], reverse=True)

        exposure = 0.0
        for column, fraction, _ in candidates:
            if exposure >= self.bet_max_exposure:
                break
            amount = max(min_bet, min(max_bet, bankroll * fraction))
            remaining = self.bet_max_exposure - exposure
            if amount / bankroll > remaining:
                amount = max(min_bet, bankroll * remaining)
            yield column, amount
            exposure += amount / bankroll
