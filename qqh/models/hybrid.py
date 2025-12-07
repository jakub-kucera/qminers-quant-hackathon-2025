"""The ensemble's probabilities with Pepe's confidence-weighted Kelly staking.

Our best model in the online rounds; the finals' round-2 submission and, with other settings,
the Bookie Rookie one.
"""

import pandas as pd

from qqh.models.ensemble import AWAY, DRAW, HOME, EnsembleModel, empty_bets
from qqh.staking import ConfidenceKelly

_PREDICTION_KEYS = (
    "ensemble_required_n",
    "ensemble_retrain_freq",
    "pythagorean_power",
    "pythagorean_regularization",
    "pythagorean_decay",
    "four_factor_required_n",
    "four_factor_decay",
    "gradient_descent_epochs",
    "gradient_descent_min_games",
    "exhaustion_b2b_penalty",
    "exhaustion_road_penalty",
    "elo_k_base",
    "elo_home_advantage",
    "elo_margin_cap",
    "poisson_regularization",
    "poisson_decay",
    "rolling_window_short",
    "rolling_window_med",
    "rolling_window_long",
)

# Finals, round 2, as committed at 12:04 on the finals day: Optuna's trial 6 on the finals'
# training data (seasons 2010-2014, in sample). Most likely the round-2 upload: replayed on
# round 2 it places no bets, as the evaluation server reported.
FINALS_ROUND_2 = dict(
    ensemble_required_n=700,
    ensemble_retrain_freq=350,
    pythagorean_power=1.2654047896737288,
    pythagorean_regularization=35,
    pythagorean_decay=0.9914605630145961,
    four_factor_required_n=400,
    four_factor_decay=0.9906944262706785,
    gradient_descent_epochs=15,
    gradient_descent_min_games=500,
    exhaustion_b2b_penalty=1.6122378907451644,
    exhaustion_road_penalty=0.3098116140922794,
    elo_k_base=25,
    elo_home_advantage=40,
    elo_margin_cap=5,
    poisson_regularization=3.9821364645005977,
    poisson_decay=0.9962778533599858,
    rolling_window_short=3,
    rolling_window_med=14,
    rolling_window_long=22,
    min_edge=0.07642150043685061,
    kelly_fraction=0.5577068347747918,
    value_multiplier=3.8955878786984246,
    confidence_threshold=0.6058143644490847,
    max_bet_fraction=0.2707878837511299,
    max_total_exposure=0.9303174396487595,
    weight_separation=0.4938394813836171,
    weight_certainty=0.4440286639439308,
    weight_form=0.3017698014588763,
    start_season=2010,
)

# Retuned later on the finals day (the last version on disk): Optuna's trial 11 on the same
# data, betting only from season 2017. Probably never submitted.
FINALS_RETUNED = dict(
    ensemble_required_n=700,
    ensemble_retrain_freq=350,
    pythagorean_power=1.0468184350832361,
    pythagorean_regularization=30,
    pythagorean_decay=0.9822964120272627,
    four_factor_required_n=400,
    four_factor_decay=0.9916597424416206,
    gradient_descent_epochs=15,
    gradient_descent_min_games=500,
    exhaustion_b2b_penalty=1.6666338402045837,
    exhaustion_road_penalty=0.3098116140922794,
    elo_k_base=20,
    elo_home_advantage=30,
    elo_margin_cap=5,
    poisson_regularization=3.7095404610906284,
    poisson_decay=0.9934047897209451,
    rolling_window_short=3,
    rolling_window_med=16,
    rolling_window_long=24,
    min_edge=0.053442535743802144,
    kelly_fraction=0.4978664986203395,
    value_multiplier=1.621010771577343,
    confidence_threshold=0.5537916773266711,
    max_bet_fraction=0.15379896883899621,
    max_total_exposure=0.7611977639797657,
    weight_separation=0.3129216888453512,
    weight_certainty=0.586798930239166,
    weight_form=0.3017698014588763,
    start_season=2017,
)

# Bookie Rookie, the second qualification task (bankroll 100, bets 1-20, a bookmaker with
# looser odds). Optuna's trial 194 on that task's training seasons, in sample: +3,031%.
BOOKIE_ROOKIE = dict(
    ensemble_required_n=700,
    ensemble_retrain_freq=150,
    pythagorean_power=1.031890264397898,
    pythagorean_regularization=30,
    pythagorean_decay=0.9893202028961737,
    four_factor_required_n=300,
    four_factor_decay=0.9917848635848201,
    gradient_descent_epochs=15,
    gradient_descent_min_games=500,
    exhaustion_b2b_penalty=1.6526715430456378,
    exhaustion_road_penalty=0.3095261468561664,
    elo_k_base=15,
    elo_home_advantage=55,
    elo_margin_cap=7,
    poisson_regularization=3.708116512060344,
    poisson_decay=0.992045263584752,
    rolling_window_short=3,
    rolling_window_med=11,
    rolling_window_long=24,
    min_edge=0.07993205778464312,
    kelly_fraction=0.3084802286209483,
    value_multiplier=1.7611869443210917,
    confidence_threshold=0.4502471850554413,
    max_bet_fraction=0.24799482880841384,
    max_total_exposure=0.8991473285385401,
    weight_separation=0.11548032865322859,
    weight_certainty=0.2437031965069546,
    weight_form=0.6408164748398169,
    start_season=None,
)


class HybridModel(EnsembleModel):
    """EnsembleModel's predictions, staked with ConfidenceKelly; no bets before `start_season`.

    Keyword arguments are the ensemble's prediction settings and ConfidenceKelly's.
    """

    def __init__(self, start_season=None, **settings):
        prediction = {k: settings.pop(k) for k in _PREDICTION_KEYS if k in settings}
        super().__init__(**prediction)
        self.staking = ConfidenceKelly(**settings)
        self.start_season = start_season

    def _ingest(self, inc: pd.DataFrame):
        for idx in inc.index:
            game = inc.loc[idx]
            self._add_game(game)
            self.staking.add_game(game)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        self._ingest(inc)
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        bets = empty_bets(opps)
        total = 0.0
        for i in opps.index:
            game = opps.loc[i]
            # Checked before predicting, which has side effects (see EnsembleModel._features).
            if self.start_season and int(game["Season"]) < self.start_season:
                continue
            probs = self._predict(game)
            if probs is None:
                continue
            if any(o <= 1.0 for o in (game["OddsH"], game["OddsA"], game["OddsD"])):
                continue
            bet = self.staking.stake(
                probs[HOME], probs[DRAW], probs[AWAY], game["OddsH"], game["OddsA"],
                game["HID"], game["AID"], bankroll, min_bet, max_bet,
            )  # fmt: skip
            if bet is None:
                continue
            column, amount = bet
            if total + amount <= bankroll * self.staking.max_total_exposure:
                bets.at[i, column] = amount
                total += amount
        return self.staking.cap_total(bets, bankroll, min_bet)
