"""Turning probabilities into stakes."""

from collections import defaultdict

import numpy as np
import scipy.optimize


class ConfidenceKelly:
    """Pepe's staking rule (Přemysl Pilař): bet only when the model is confident, then a
    Kelly-like stake grown by how far the edge clears its minimum.

    Confidence mixes three things, with weights normalised to sum to 1:
    - separation: the gap between the two most likely outcomes,
    - certainty: 1 - entropy / ln 3,
    - form consistency: 1 / (1 + var(home form) + var(away form)) over the last 14 results.

    Only the better of home and away is bet (draws never), with stake
    kelly_fraction x edge x bankroll x min(value_multiplier, edge / min_edge) x confidence,
    capped per bet and, for the day's bets together, by a share of the bankroll.
    """

    form_window = 14

    def __init__(
        self,
        min_edge,
        kelly_fraction,
        value_multiplier,
        confidence_threshold,
        max_bet_fraction,
        max_total_exposure,
        weight_separation,
        weight_certainty,
        weight_form,
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.value_multiplier = value_multiplier
        self.confidence_threshold = confidence_threshold
        self.max_bet_fraction = max_bet_fraction
        self.max_total_exposure = max_total_exposure
        total = weight_separation + weight_certainty + weight_form
        self.weight_separation = weight_separation / total
        self.weight_certainty = weight_certainty / total
        self.weight_form = weight_form / total
        self.form = defaultdict(list)  # team -> its last results (1 win, 0.5 draw, 0 loss)

    def add_game(self, game):
        if game["H"]:
            home, away = 1.0, 0.0
        elif game["A"]:
            home, away = 0.0, 1.0
        else:
            home, away = 0.5, 0.5
        for team_id, result in ((game["HID"], home), (game["AID"], away)):
            self.form[team_id].append(result)
            if len(self.form[team_id]) > self.form_window:
                self.form[team_id].pop(0)

    def confidence(self, probs, home_id, away_id):
        """probs is [P(home), P(draw), P(away)]."""
        ordered = sorted(probs, reverse=True)
        separation = ordered[0] - ordered[1]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        certainty = 1.0 - (entropy / np.log(3))
        home_form = self.form.get(home_id, [])
        away_form = self.form.get(away_id, [])
        home_var = np.var(home_form) if len(home_form) > 1 else 0.0
        away_var = np.var(away_form) if len(away_form) > 1 else 0.0
        consistency = 1.0 / (1.0 + home_var + away_var)
        return (
            self.weight_separation * separation
            + self.weight_certainty * certainty
            + self.weight_form * consistency
        )

    def stake(self, prob_home, prob_draw, prob_away, odds_home, odds_away, home_id, away_id,
              bankroll, min_bet, max_bet):  # fmt: skip
        """(column, amount) for one game, or None. Doesn't apply the day's exposure cap."""
        edge_home = prob_home * odds_home - 1
        edge_away = prob_away * odds_away - 1
        confidence = self.confidence([prob_home, prob_draw, prob_away], home_id, away_id)
        if confidence < self.confidence_threshold:
            return None
        column, edge = ("BetH", edge_home) if edge_home > edge_away else ("BetA", edge_away)
        if not edge > self.min_edge:
            return None
        value = edge / self.min_edge
        amount = (
            self.kelly_fraction * edge * bankroll * min(self.value_multiplier, value) * confidence
        )
        amount = max(min_bet, min(amount, max_bet, bankroll * self.max_bet_fraction))
        return column, np.round(amount)

    def cap_total(self, bets, bankroll, min_bet):
        """Scale the day's bets down if together they exceed the exposure cap."""
        total = bets.sum().sum()
        if total > bankroll * self.max_total_exposure:
            bets = np.round(bets * ((bankroll * self.max_total_exposure) / total))
            bets[bets < min_bet] = 0
        return bets


def kelly_simultaneous(probs, odds, fraction=0.3):
    """Kelly stakes for one game's mutually exclusive outcomes, as bankroll fractions.

    Maximises the expected log of wealth over all outcomes at once (SLSQP), so it can back
    more than one outcome; then scales by `fraction`. From Marek Vedral's model.
    """
    p = np.array(probs)
    b = np.array(odds)
    if np.max(p * b) <= 1.0:
        return np.zeros(len(b))

    def neg_log_growth(f):
        wealth = 1.0 - np.sum(f) + (f * b)
        return -np.sum(p * np.log(np.maximum(wealth, 1e-10)))

    result = scipy.optimize.minimize(
        neg_log_growth,
        np.zeros(len(b)),
        method="SLSQP",
        bounds=[(0, 1)] * len(b),
        constraints={"type": "ineq", "fun": lambda f: 1.0 - np.sum(f)},
    )
    stakes = result.x * fraction
    stakes[stakes < 0.0001] = 0.0
    return stakes
