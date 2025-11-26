import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import scipy

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
    # EV = Probability * Odds. If EV <= 1, Kelly bet is always 0.
    expected_values = p * b

    # If no value exists anywhere, don't waste time optimizing
    if np.max(expected_values) <= 1.0:
        return {
            "bet_fractions": np.zeros(3),
            "bet_amounts": np.zeros(3),
            "expected_growth": 0.0,
        }

    # 2. Define the Negative Expected Log Growth (to minimize)
    def neg_log_growth(f):
        # f is an array of bet fractions [f_H, f_D, f_A]
        # Wealth if outcome i hits: 1 - sum(bets) + (bet_on_i * odds_i)
        # Note: We enforce sum(f) < 1 via constraints later, but add epsilon for safety
        current_wealths = 1.0 - np.sum(f) + (f * b)

        # Log growth = Sum( p_i * ln(wealth_i) )
        # We assume utility of ruin is -infinity, so clamp wealth > 0
        safe_wealths = np.maximum(current_wealths, 1e-10)
        return -np.sum(p * np.log(safe_wealths))

    # 3. Constraints & Bounds
    # Constraint: Sum of bets < 1.0 (cannot bet more than bankroll)
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

    # 5. Apply Fractional Kelly (Risk Management)
    # We scale the optimal fractions down.
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
        self.model.load_state_dict(torch.load(MODEL_FILENAME))
        self.model.eval()
        self.scaler_mean = np.load(SCALER_MEAN_FILENAME)
        self.scaler_std = np.load(SCALER_VAR_FILENAME)
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
