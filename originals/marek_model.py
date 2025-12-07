import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy
import base64
import io

PRED_MIN_MATCHES = 500
SEASON_MIN_MATCHES = 100
TEAM_EMBED_LEN = 15
FEATURE_LEN = TEAM_EMBED_LEN + 12
LAST_MATCH_NUM = 5

MODEL_FILENAME = "model.json"
SCALER_MEAN_FILENAME = "scaler_mean.npy"
SCALER_VAR_FILENAME = "scaler_var.npy"

ELO_HOME_ADVANTAGE = 30
ELO_K = 25
INIT_ELO_RATING = 1000

# upper bound on the number of teams
TEAM_NUM = 50
UB = 10000
LAST_PRED_WINDOW = 20

# pasted values from files for the evaluation system
SCALER_MEAN = [
    7.85733088e-02,
    8.68649645e-03,
    6.98868123e-02,
    4.34324822e-02,
    -1.93471966e-02,
    6.27796789e-02,
    -7.76520137e-03,
    -9.47617794e-03,
    1.71097657e-03,
    -5.25138194e-02,
    -7.77836273e-02,
    -5.18754936e00,
    4.86970255e-03,
    5.81732035e-02,
    1.88602264e-01,
    5.42254686e-01,
    3.33937067e-01,
    1.02223619e-01,
    8.27934445e-01,
    1.06394067e03,
    1.06605489e03,
    -2.11421270e00,
    5.35541347e-01,
    4.64458653e-01,
    -2.27057087e-01,
    -1.11828466e00,
    -1.01894300e01,
]

SCALER_VAR = [
    9.52044455e00,
    9.21368555e00,
    1.34926685e01,
    3.97056933e00,
    3.83864742e00,
    5.31764205e00,
    1.51957231e00,
    1.51300921e00,
    2.18465110e00,
    4.54260346e01,
    4.82714737e01,
    2.85532750e01,
    3.82178713e00,
    1.23772854e01,
    5.77020635e01,
    1.29210463e-01,
    1.02884131e-01,
    9.46080423e-02,
    2.13477085e-01,
    9.44456977e01,
    9.43427509e01,
    1.32116467e02,
    1.65984125e-01,
    1.65984125e-01,
    3.07389835e00,
    2.94200095e00,
    9.57946374e00,
]

# base64 encoded model params
NN_PARAMS = """
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
"""


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
        # convert the bookmaker probs
        market_logits = x[:, -3:]
        return self.net(x) + market_logits


def load_base64_params():
    """Load the model parameters in the base64 string"""
    raw = base64.b64decode(NN_PARAMS)
    buffer = io.BytesIO(raw)
    return buffer


def odds_to_probs(input_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the odds to probabilities in the DF"""
    df = input_df.copy()

    # convert the odds to probabilities
    df["ProbH"] = 1 / df["OddsH"]
    df["ProbA"] = 1 / df["OddsA"]
    df["ProbD"] = 1 / df["OddsD"]

    # replace the infinity in case any prob is 0
    df["ProbH"] = df["ProbH"].replace(np.inf, 0)
    df["ProbA"] = df["ProbA"].replace(np.inf, 0)
    df["ProbD"] = df["ProbD"].replace(np.inf, 0)

    # de-vig the probabilities
    overround = df["ProbA"] + df["ProbH"] + df["ProbD"]

    df["ProbH"] = df["ProbH"] / overround
    df["ProbA"] = df["ProbA"] / overround
    df["ProbD"] = df["ProbD"] / overround

    df.fillna(0, inplace=True)

    return df


def add_corsi_fenwick(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add the Fenwick and Corsi statisctics"""
    df = summary_df.copy()
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


def elo_get_expected(elo: np.ndarray, hid: int, aid: int):
    """Get expected ELO"""
    R_home = elo[hid]
    R_away = elo[aid]

    exp_home = 1 / (1 + 10 ** (-(R_home + ELO_HOME_ADVANTAGE - R_away) / 400))

    return exp_home, 1 - exp_home


def update_elo(
    cur_elo: np.ndarray, hid: int, aid: int, h_goals: int, a_goals: int, res: int
):
    """Update the ELO rating"""
    exp_home, _ = elo_get_expected(cur_elo, hid, aid)

    if res == 0:
        # home win
        S_home, S_away = 1, 0
    elif res == 1:
        # away win
        S_home, S_away = 0, 1
    else:
        # draw
        S_home, S_away = 0.5, 0.5

    # Goal differential modifier
    margin = abs(h_goals - a_goals)
    goal_mult = np.log(1 + margin)

    # Adjusted K
    K_adj = ELO_K * goal_mult

    new_R_home = cur_elo[hid] + K_adj * (S_home - exp_home)
    new_R_away = cur_elo[aid] + K_adj * (S_away - (1 - exp_home))

    # return updated elo
    new_elo = cur_elo.copy()
    new_elo[hid] = new_R_home
    new_elo[aid] = new_R_away

    return new_elo


def calculate_kelly_stakes(model_probs, market_odds, bankroll, fractional_kelly=0.3):
    """
    Calculates optimal bets for Mutually Exclusive outcomes (1x2).

    Args:
        model_probs (list): [Prob_Home, Prob_Draw, Prob_Away] from your NN.
        market_odds (list): [Odds_Home, Odds_Draw, Odds_Away] from Bookie (Decimal).
        bankroll (float): Total money available.
        fractional_kelly (float): Safety factor (e.g., 0.3 for 30% Kelly).
                                  Professional bettors rarely use full Kelly (1.0).

    Returns:
        dict: Recommended bets in cash and percentage.
    """
    p = np.array(model_probs)
    b = np.array(market_odds)

    # 1. Check for value (Expected Value > 1.0)
    expected_values = p * b

    # ff no value exists anywhere, don't optimize
    if np.max(expected_values) <= 1.0:
        return {
            "bet_fractions": np.zeros(3),
            "bet_amounts": np.zeros(3),
            "expected_growth": 0.0,
        }

    # 2. Define the Negative Expected Log Growth (to minimize)
    def neg_log_growth(f):
        current_wealths = 1.0 - np.sum(f) + (f * b)
        safe_wealths = np.maximum(current_wealths, 1e-10)
        return -np.sum(p * np.log(safe_wealths))

    # 3. Constraints & Bounds
    constraints = {"type": "ineq", "fun": lambda f: 1.0 - np.sum(f)}
    # Bounds: Each bet must be 0 <= f <= 1
    bounds = [(0, 1) for _ in range(3)]

    # 4. Optimize
    # Start guess: bet 0 on everything
    initial_guess = np.zeros(3)

    result = scipy.optimize.minimize(
        neg_log_growth,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_f = result.x

    # 5. Apply Fractional Kelly
    final_f = optimal_f * fractional_kelly

    # Zero out tiny bets (numerical noise)
    final_f[final_f < 0.0001] = 0.0

    return {
        "bet_fractions": final_f,  # Vector [f_H, f_D, f_A]
        "bet_amounts": final_f * bankroll,
        "raw_model_ev": expected_values,
    }


def get_team_stats(
    season: int,
    team_id: int,
    df: pd.DataFrame,
    cur_date: pd.Timestamp,
    last_match_lim: int = -1,
):
    """Get the feature vector from past matches"""

    if season < 0:
        # ignore the season
        home_matches = df[(df["HID"] == team_id)].copy()
        away_matches = df[(df["AID"] == team_id)].copy()
    else:
        home_matches = df[(df["HID"] == team_id) & (df["Season"] == season)].copy()
        away_matches = df[(df["AID"] == team_id) & (df["Season"] == season)].copy()

    if last_match_lim > 0:
        home_matches = home_matches.tail(last_match_lim)
        away_matches = away_matches.tail(last_match_lim)

    home_matches["Home"] = True
    away_matches["Home"] = False

    # get the statistics
    games_played = len(home_matches) + len(away_matches)

    if games_played == 0:
        return np.zeros(TEAM_EMBED_LEN)

    goals_for = home_matches["HS"].sum() + away_matches["AS"].sum()
    goals_against = home_matches["AS"].sum() + away_matches["HS"].sum()
    goal_diff = goals_for - goals_against

    goals_for_ppg = home_matches["H_PPG"].sum() + away_matches["A_PPG"].sum()
    goals_against_ppg = home_matches["A_PPG"].sum() + away_matches["H_PPG"].sum()
    goals_ppg_diff = goals_for_ppg - goals_against_ppg

    goals_for_shg = home_matches["H_SHG"].sum() + away_matches["A_SHG"].sum()
    goals_against_shg = home_matches["A_SHG"].sum() + away_matches["H_SHG"].sum()
    goals_shg_diff = goals_for_shg - goals_against_shg

    fenwick_total = home_matches["H_FEN"].sum() + away_matches["A_FEN"].sum()
    corsi_total = home_matches["H_COR"].sum() + away_matches["A_COR"].sum()

    # days since the last match
    home_last = np.inf
    away_last = np.inf
    if len(home_matches) > 0:
        home_last = (
            pd.Timestamp(cur_date) - pd.Timestamp(home_matches["Date"].iloc[-1])
        ).days
    if len(away_matches) > 0:
        home_last = (
            pd.Timestamp(cur_date) - pd.Timestamp(away_matches["Date"].iloc[-1])
        ).days
    days_last = min(home_last, away_last)

    # penalties
    major_penalties = home_matches["H_MAJ"].sum() + away_matches["A_MAJ"].sum()
    minor_penalties = home_matches["H_PEN"].sum() + away_matches["A_PEN"].sum()
    penalty_time = home_matches["H_PIM"].sum() + away_matches["A_PIM"].sum()

    ret = np.array(
        (
            goals_for,
            goals_against,
            goal_diff,
            goals_for_ppg,
            goals_against_ppg,
            goals_ppg_diff,
            goals_for_shg,
            goals_against_shg,
            goals_shg_diff,
            fenwick_total,
            corsi_total,
            days_last,
            major_penalties,
            minor_penalties,
            penalty_time,
        )
    )

    return ret


class Model:

    def __init__(self) -> None:
        self.past_matches = None
        self.elo = np.ones(TEAM_NUM) * INIT_ELO_RATING
        self.model = MatchPredictor()
        self.model.load_state_dict(torch.load(load_base64_params()))
        self.model.eval()
        self.scaler_mean = SCALER_MEAN
        self.scaler_std = SCALER_VAR
        self.eps = 1e-9

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        # replace NaN values
        inc.fillna(0, inplace=True)
        # add data to past matches
        inc = add_corsi_fenwick(odds_to_probs(inc))

        # update the ELO rating
        for inc_idx in range(len(inc)):
            row = inc.iloc[inc_idx]
            if row["H"]:
                res_int = 0
            elif row["A"]:
                res_int = 1
            else:
                res_int = 2

            self.elo = update_elo(
                self.elo, row["HID"], row["AID"], row["HS"], row["AS"], res_int
            )

        # save the incremental data
        if self.past_matches is None:
            self.past_matches = inc
        else:
            self.past_matches = pd.concat((self.past_matches, inc))

        # check if we have enough data
        if len(self.past_matches) < PRED_MIN_MATCHES:
            return pd.DataFrame()

        # de-vig the opps odds
        opps = odds_to_probs(opps)

        bets = np.zeros((len(opps), 3))
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        cur_bankroll = summary.iloc[0]["Bankroll"]

        # make the bets
        for op_idx in range(len(opps)):
            row = opps.iloc[op_idx]

            if row["BetH"] + row["BetA"] + row["BetD"] > 0:
                # already bet
                continue

            # build the feature vector
            df = self.past_matches.copy()

            # get data for this season
            cur_season_df = df[
                (df["Season"] == row["Season"]) & (df["Date"] < row["Date"])
            ].reset_index()

            if len(cur_season_df) < SEASON_MIN_MATCHES:
                # get the last season matches too
                last_season_df = df[df["Season"] == (row["Season"] - 1)]

                # get the most recent matches
                cur_season_df = pd.concat(
                    (last_season_df.tail(SEASON_MIN_MATCHES), cur_season_df)
                )

            # get the statistics for both teams
            home_stats = get_team_stats(
                -1, row["HID"], cur_season_df, row["Date"], LAST_MATCH_NUM
            )
            away_stats = get_team_stats(
                -1, row["AID"], cur_season_df, row["Date"], LAST_MATCH_NUM
            )

            diff = home_stats - away_stats

            book_probs = np.array([row["ProbH"], row["ProbA"], row["ProbD"]])

            # add the bookmaker odds
            diff = np.append(diff, book_probs)

            # add market entropy
            diff = np.append(diff, -np.sum(book_probs * np.log(book_probs + self.eps)))

            # add elo rating for both teams
            elo_h = self.elo[row["HID"]]
            elo_a = self.elo[row["AID"]]
            elo_h_expected, elo_a_expected = elo_get_expected(
                self.elo, row["HID"], row["AID"]
            )
            diff = np.append(
                diff, [elo_h, elo_a, elo_h - elo_a, elo_h_expected, elo_a_expected]
            )

            # add the logits at the end
            f_logits = np.log(book_probs + self.eps) - np.log(1 - book_probs + self.eps)
            diff = np.append(diff, f_logits)

            # scale the feature vector
            diff = (diff - self.scaler_mean) / self.scaler_std
            diff = np.atleast_2d(diff)

            # make the prediction
            res = torch.softmax(self.model(torch.from_numpy(diff).float()), dim=1)
            raw_odds = [row["OddsH"], row["OddsA"], row["OddsD"]]
            decision = calculate_kelly_stakes(
                res.detach().numpy(), raw_odds, cur_bankroll
            )

            # check that the amounts are within range
            dec = np.array(decision["bet_amounts"])
            for i in range(len(dec)):
                if dec[i] > 0:
                    dec[i] = np.clip(dec[i], min_bet, max_bet)

            bets[op_idx, :] = dec

            # update the bankroll for future bets
            cur_bankroll -= np.sum(dec)

        bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetA", "BetD"], index=opps.index
        )
        return bets
