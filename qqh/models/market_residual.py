"""Marek Vedral's model: a small network that corrects the bookmaker's probabilities.

Features for a game are the home-minus-away difference of 15 team stats over each team's last
5 home and 5 away games, the bookmaker's probabilities (margin removed), the market's entropy,
an Elo and the bookmaker's log-odds. The network's output is added to those log-odds
(a residual on the market) and softmaxed into home/away/draw probabilities. Stakes come from
`kelly_simultaneous` at 30% Kelly.

The network was trained offline on the qualification data (seasons 1989-2010, a random 2000
games held out); only its weights are here. Known issues, kept as submitted
(see docs/postmortem.md):
- the inputs are divided by the scaler's variance, not its standard deviation;
- the residual is added to the *standardised* log-odds, not the raw ones;
- "days since the last game" only ever sees the team's last away game;
- "Corsi" and "Fenwick" count the opponent's shots on goal and the saves, not shot attempts.
"""

import base64
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from qqh.models._market_residual_weights import WEIGHTS
from qqh.staking import kelly_simultaneous

PRED_MIN_MATCHES = 500
SEASON_MIN_MATCHES = 100
TEAM_EMBED_LEN = 15
FEATURE_LEN = TEAM_EMBED_LEN + 12
LAST_MATCH_NUM = 5
ELO_HOME_ADVANTAGE = 30
ELO_K = 25
INIT_ELO_RATING = 1000
TEAM_NUM = 50

# Standardisation of the 27 inputs, from the training run.
SCALER_MEAN = [
    0.0785733088,
    0.00868649645,
    0.0698868123,
    0.0434324822,
    -0.0193471966,
    0.0627796789,
    -0.00776520137,
    -0.00947617794,
    0.00171097657,
    -0.0525138194,
    -0.0777836273,
    -5.18754936,
    0.00486970255,
    0.0581732035,
    0.188602264,
    0.542254686,
    0.333937067,
    0.102223619,
    0.827934445,
    1063.94067,
    1066.05489,
    -2.1142127,
    0.535541347,
    0.464458653,
    -0.227057087,
    -1.11828466,
    -10.18943,
]

SCALER_VAR = [
    9.52044455,
    9.21368555,
    13.4926685,
    3.97056933,
    3.83864742,
    5.31764205,
    1.51957231,
    1.51300921,
    2.1846511,
    45.4260346,
    48.2714737,
    28.553275,
    3.82178713,
    12.3772854,
    57.7020635,
    0.129210463,
    0.102884131,
    0.0946080423,
    0.213477085,
    94.4456977,
    94.3427509,
    132.116467,
    0.165984125,
    0.165984125,
    3.07389835,
    2.94200095,
    9.57946374,
]


class MatchPredictor(nn.Module):
    def __init__(self, input_dim=FEATURE_LEN, hidden1=64, hidden2=32, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        market_logits = x[:, -3:]
        return self.net(x) + market_logits


def odds_to_probs(input_df: pd.DataFrame) -> pd.DataFrame:
    """Add ProbH/ProbA/ProbD: the bookmaker's probabilities with its margin removed."""
    df = input_df.copy()
    df["ProbH"] = 1 / df["OddsH"]
    df["ProbA"] = 1 / df["OddsA"]
    df["ProbD"] = 1 / df["OddsD"]
    df["ProbH"] = df["ProbH"].replace(np.inf, 0)
    df["ProbA"] = df["ProbA"].replace(np.inf, 0)
    df["ProbD"] = df["ProbD"].replace(np.inf, 0)
    overround = df["ProbA"] + df["ProbH"] + df["ProbD"]
    df["ProbH"] = df["ProbH"] / overround
    df["ProbA"] = df["ProbA"] / overround
    df["ProbD"] = df["ProbD"] / overround
    df.fillna(0, inplace=True)
    return df


def add_corsi_fenwick(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    home_shot_attempts = df["A_SOG"]
    away_shot_attempts = df["H_SOG"]
    home_missed_shots = home_shot_attempts - df["HS"]
    away_missed_shots = away_shot_attempts - df["AS"]
    home_blocks = df["A_BLK_S"]
    away_blocks = df["H_BLK_S"]
    df["H_COR"] = (home_shot_attempts + home_missed_shots + home_blocks) - (
        away_shot_attempts + away_missed_shots + away_blocks
    )
    df["A_COR"] = -df["H_COR"]
    df["H_FEN"] = (home_shot_attempts + home_missed_shots) - (
        away_shot_attempts + away_missed_shots
    )
    df["A_FEN"] = -df["H_FEN"]
    return df


def elo_expected(elo: np.ndarray, hid: int, aid: int):
    exp_home = 1 / (1 + 10 ** (-(elo[hid] + ELO_HOME_ADVANTAGE - elo[aid]) / 400))
    return exp_home, 1 - exp_home


def update_elo(elo: np.ndarray, hid: int, aid: int, h_goals: int, a_goals: int, res: int):
    """Elo with K scaled by ln(1 + margin); res is 0 home win, 1 away win, 2 draw."""
    exp_home, _ = elo_expected(elo, hid, aid)
    if res == 0:
        s_home, s_away = 1, 0
    elif res == 1:
        s_home, s_away = 0, 1
    else:
        s_home, s_away = 0.5, 0.5
    k = ELO_K * np.log(1 + abs(h_goals - a_goals))
    new_home = elo[hid] + k * (s_home - exp_home)
    new_away = elo[aid] + k * (s_away - (1 - exp_home))
    new_elo = elo.copy()
    new_elo[hid] = new_home
    new_elo[aid] = new_away
    return new_elo


def team_stats(team_id: int, df: pd.DataFrame, date: pd.Timestamp, last: int) -> np.ndarray:
    """15 stats over the team's last `last` home and last `last` away games in `df`."""
    home = df[df["HID"] == team_id].copy().tail(last)
    away = df[df["AID"] == team_id].copy().tail(last)
    if len(home) + len(away) == 0:
        return np.zeros(TEAM_EMBED_LEN)
    goals_for = home["HS"].sum() + away["AS"].sum()
    goals_against = home["AS"].sum() + away["HS"].sum()
    ppg_for = home["H_PPG"].sum() + away["A_PPG"].sum()
    ppg_against = home["A_PPG"].sum() + away["H_PPG"].sum()
    shg_for = home["H_SHG"].sum() + away["A_SHG"].sum()
    shg_against = home["A_SHG"].sum() + away["H_SHG"].sum()
    home_last = np.inf
    away_last = np.inf
    if len(home) > 0:
        home_last = (pd.Timestamp(date) - pd.Timestamp(home["Date"].iloc[-1])).days
    if len(away) > 0:
        home_last = (pd.Timestamp(date) - pd.Timestamp(away["Date"].iloc[-1])).days
    return np.array(
        (
            goals_for,
            goals_against,
            goals_for - goals_against,
            ppg_for,
            ppg_against,
            ppg_for - ppg_against,
            shg_for,
            shg_against,
            shg_for - shg_against,
            home["H_FEN"].sum() + away["A_FEN"].sum(),
            home["H_COR"].sum() + away["A_COR"].sum(),
            min(home_last, away_last),
            home["H_MAJ"].sum() + away["A_MAJ"].sum(),
            home["H_PEN"].sum() + away["A_PEN"].sum(),
            home["H_PIM"].sum() + away["A_PIM"].sum(),
        )
    )


class MarketResidualNet:
    eps = 1e-9

    def __init__(self, kelly_fraction=0.3):
        self.kelly_fraction = kelly_fraction
        self.past_matches = None
        self.elo = np.ones(TEAM_NUM) * INIT_ELO_RATING
        self.model = MatchPredictor()
        weights = io.BytesIO(base64.b64decode("".join(WEIGHTS)))
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def _ingest(self, inc: pd.DataFrame):
        inc = inc.fillna(0)
        inc = add_corsi_fenwick(odds_to_probs(inc))
        for i in range(len(inc)):
            game = inc.iloc[i]
            result = 0 if game["H"] else 1 if game["A"] else 2
            self.elo = update_elo(
                self.elo, game["HID"], game["AID"], game["HS"], game["AS"], result
            )
        if self.past_matches is None:
            self.past_matches = inc
        else:
            self.past_matches = pd.concat((self.past_matches, inc))

    def _features(self, game) -> np.ndarray:
        df = self.past_matches.copy()
        season = df[(df["Season"] == game["Season"]) & (df["Date"] < game["Date"])].reset_index()
        if len(season) < SEASON_MIN_MATCHES:
            previous = df[df["Season"] == (game["Season"] - 1)]
            season = pd.concat((previous.tail(SEASON_MIN_MATCHES), season))
        home = team_stats(game["HID"], season, game["Date"], LAST_MATCH_NUM)
        away = team_stats(game["AID"], season, game["Date"], LAST_MATCH_NUM)
        book = np.array([game["ProbH"], game["ProbA"], game["ProbD"]])
        elo_home, elo_away = self.elo[game["HID"]], self.elo[game["AID"]]
        expected_home, expected_away = elo_expected(self.elo, game["HID"], game["AID"])
        x = home - away
        x = np.append(x, book)
        x = np.append(x, -np.sum(book * np.log(book + self.eps)))
        x = np.append(x, [elo_home, elo_away, elo_home - elo_away, expected_home, expected_away])
        x = np.append(x, np.log(book + self.eps) - np.log(1 - book + self.eps))
        return (x - SCALER_MEAN) / SCALER_VAR

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        self._ingest(inc)
        if len(self.past_matches) < PRED_MIN_MATCHES:
            return pd.DataFrame()
        opps = odds_to_probs(opps)
        bets = np.zeros((len(opps), 3))
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        for row in range(len(opps)):
            game = opps.iloc[row]
            if game["BetH"] + game["BetA"] + game["BetD"] > 0:
                continue  # already bet on an earlier day
            x = np.atleast_2d(self._features(game))
            probs = torch.softmax(self.model(torch.from_numpy(x).float()), dim=1)
            odds = [game["OddsH"], game["OddsA"], game["OddsD"]]
            stakes = (
                kelly_simultaneous(probs.detach().numpy().ravel(), odds, self.kelly_fraction)
                * bankroll
            )
            for i in range(len(stakes)):
                if stakes[i] > 0:
                    stakes[i] = np.clip(stakes[i], min_bet, max_bet)
            bets[row, :] = stakes
            bankroll -= np.sum(stakes)
        return pd.DataFrame(data=bets, columns=["BetH", "BetA", "BetD"], index=opps.index)
