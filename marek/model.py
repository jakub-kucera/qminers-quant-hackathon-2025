import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PRED_MIN_MATCHES = 500
SEASON_MIN_MATCHES = 100
TEAM_EMBED_LEN = 15
FEATURE_LEN = TEAM_EMBED_LEN + 8
LAST_MATCH_NUM = 5
MODEL_FILENAME = "model.json"

ELO_HOME_ADVANTAGE = 30
ELO_K = 25
INIT_ELO_RATING = 1000

# upper bound on the number of teams
TEAM_NUM = 50


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
        self.model = XGBClassifier()
        self.model.load_model(MODEL_FILENAME)

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
                self.elo,
                row["HID"],
                row["AID"],
                row["HS"],
                row["AS"],
                res_int
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

            # add the odds
            diff = np.append(diff, [row["ProbH"], row["ProbA"], row["ProbD"]])

            # append data from the bet opportunity
            elo_h = self.elo[row["HID"]]
            elo_a = self.elo[row["AID"]]
            elo_h_expected, elo_a_expected = elo_get_expected(
                self.elo, row["HID"], row["AID"]
            )
            diff = np.atleast_2d(
                np.append(
                    diff, [elo_h, elo_a, elo_h - elo_a, elo_h_expected, elo_a_expected]
                )
            )

            # make the prediction
            res = self.model.predict(diff)[0]

            bets[op_idx, res] = min_bet

        bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetA", "BetD"], index=opps.index
        )
        return bets
