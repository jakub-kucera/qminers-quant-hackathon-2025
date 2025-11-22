import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim


def inverse_sigmoid(x):
    """Convert probability to log-odds"""
    x = max(0.001, min(0.999, x))  # Clamp to avoid log(0) or log(negative)
    return math.log(x / (1 - x))


def sigmoid(x):
    """Convert log-odds to probability"""
    return 1 / (1 + math.exp(-x))


class Exhaustion:
    """Tracks team rest days and fatigue with hockey-specific factors"""
    def __init__(self, b2b_penalty=1.5, one_day_penalty=0.8, road_trip_penalty=0.5):
        """
        Args:
            b2b_penalty: Penalty factor for back-to-back games (0 days rest)
            one_day_penalty: Penalty factor for 1 day rest
            road_trip_penalty: Penalty factor for long road trips (3+ away games)
        """
        self.b2b_penalty = b2b_penalty
        self.one_day_penalty = one_day_penalty
        self.road_trip_penalty = road_trip_penalty
        
        self.team_rest = {}
        self.team_games = defaultdict(list)  # Track game history for road trip detection
        
    def add_game(self, current):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        
        self.team_rest[home_id] = date
        self.team_rest[away_id] = date
        
        # Track game location for road trip analysis
        self.team_games[home_id].append({'date': date, 'is_home': True})
        self.team_games[away_id].append({'date': date, 'is_home': False})
        
        # Keep only last 20 games for efficiency
        if len(self.team_games[home_id]) > 20:
            self.team_games[home_id] = self.team_games[home_id][-20:]
        if len(self.team_games[away_id]) > 20:
            self.team_games[away_id] = self.team_games[away_id][-20:]

    def _get_road_trip_factor(self, team_id, current_date):
        """Calculate road trip fatigue - 3+ consecutive away games increases fatigue"""
        if team_id not in self.team_games or len(self.team_games[team_id]) < 2:
            return 0.0
        
        # Count consecutive away games
        consecutive_away = 0
        for game in reversed(self.team_games[team_id][-10:]):
            if (current_date - game['date']).days <= 14:  # Within last 2 weeks
                if not game['is_home']:
                    consecutive_away += 1
                else:
                    break
        
        # Penalize teams on long road trips (3+ away games)
        if consecutive_away >= 3:
            return -self.road_trip_penalty
        elif consecutive_away >= 2:
            return -self.road_trip_penalty * 0.6
        return 0.0

    def get_input_data(self, home_id, away_id, date):
        if home_id not in self.team_rest or away_id not in self.team_rest:
            return None
        
        home_days = (date - self.team_rest[home_id]).days
        away_days = (date - self.team_rest[away_id]).days
        
        # Back-to-back penalty (major fatigue factor in NHL)
        factor = 0.0
        if home_days == 0:  # Back-to-back for home team
            factor += self.b2b_penalty
        elif home_days == 1:
            factor += self.one_day_penalty
            
        if away_days == 0:  # Back-to-back for away team
            factor -= self.b2b_penalty
        elif away_days == 1:
            factor -= self.one_day_penalty
        
        # Road trip fatigue
        home_road_factor = self._get_road_trip_factor(home_id, date)
        away_road_factor = self._get_road_trip_factor(away_id, date)
        
        factor += home_road_factor - away_road_factor
        
        return [factor]


class HockeyFourFactor:
    """Four-factor model adapted for ice hockey statistics"""
    def __init__(self, lr_required_n=500, daily_decay=0.994):
        """
        Args:
            lr_required_n: Minimum number of games required before making predictions
            daily_decay: Daily decay factor for time-weighted statistics (0.994 = ~99.4% per day)
        """
        self.lr_required_n = lr_required_n
        self.daily_decay = daily_decay
        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)
        self.past_pred = []
        self.lr = None
        self.lr_retrain = 0

    def _get_stats(self, date, stats):
        totals = {
            'ShotsOnGoal': 0,
            'Goals': 0,
            'Penalties': 0,
            'PowerPlayGoals': 0,
            'Saves': 0,
            'FaceoffsWon': 0,
            'Hits': 0,
            'Blocks': 0,
            'BlockedShots': 0,
            'Weight': 0
        }
        
        for stat in stats:
            weight = self.daily_decay ** abs((date - stat['Date']).days)
            
            totals['ShotsOnGoal'] += stat['ShotsOnGoal'] * weight
            totals['Goals'] += stat['Goals'] * weight
            totals['Penalties'] += stat['Penalties'] * weight
            totals['PowerPlayGoals'] += stat['PowerPlayGoals'] * weight
            totals['Saves'] += stat['Saves'] * weight
            totals['FaceoffsWon'] += stat['FaceoffsWon'] * weight
            totals['Hits'] += stat['Hits'] * weight
            totals['Blocks'] += stat['Blocks'] * weight
            totals['BlockedShots'] += stat['BlockedShots'] * weight
            totals['Weight'] += weight
        
        return totals

    def add_game(self, current):
        val1 = self._get_team_input_data(current['HID'], current['Date'])
        val2 = self._get_team_input_data(current['AID'], current['Date'])
        
        if val1 is not None and not any(math.isnan(v) or math.isinf(v) for v in val1):
            self.past_pred.append([*val1, current['H']])
            self.lr_retrain -= 1
        
        if val2 is not None and not any(math.isnan(v) or math.isinf(v) for v in val2):
            self.past_pred.append([*val2, current['A']])
            self.lr_retrain -= 1
        
        # Add home team stats - handle NaN by filling with 0
        self.team_stats_average[current['HID']].append({
            'Date': current['Date'],
            'ShotsOnGoal': current.get('H_SOG', 0) if not pd.isna(current.get('H_SOG', 0)) else 0,
            'Goals': current.get('HS', 0) if not pd.isna(current.get('HS', 0)) else 0,
            'Penalties': current.get('H_PEN', 0) if not pd.isna(current.get('H_PEN', 0)) else 0,
            'PowerPlayGoals': current.get('H_PPG', 0) if not pd.isna(current.get('H_PPG', 0)) else 0,
            'Saves': current.get('H_SV', 0) if not pd.isna(current.get('H_SV', 0)) else 0,
            'FaceoffsWon': current.get('H_FO', 0) if not pd.isna(current.get('H_FO', 0)) else 0,
            'Hits': current.get('H_HIT', 0) if not pd.isna(current.get('H_HIT', 0)) else 0,
            'Blocks': current.get('H_BLK', 0) if not pd.isna(current.get('H_BLK', 0)) else 0,
            'BlockedShots': current.get('H_BLK_S', 0) if not pd.isna(current.get('H_BLK_S', 0)) else 0
        })
        
        # Add away team stats - handle NaN by filling with 0
        self.team_stats_average[current['AID']].append({
            'Date': current['Date'],
            'ShotsOnGoal': current.get('A_SOG', 0) if not pd.isna(current.get('A_SOG', 0)) else 0,
            'Goals': current.get('AS', 0) if not pd.isna(current.get('AS', 0)) else 0,
            'Penalties': current.get('A_PEN', 0) if not pd.isna(current.get('A_PEN', 0)) else 0,
            'PowerPlayGoals': current.get('A_PPG', 0) if not pd.isna(current.get('A_PPG', 0)) else 0,
            'Saves': current.get('A_SV', 0) if not pd.isna(current.get('A_SV', 0)) else 0,
            'FaceoffsWon': current.get('A_FO', 0) if not pd.isna(current.get('A_FO', 0)) else 0,
            'Hits': current.get('A_HIT', 0) if not pd.isna(current.get('A_HIT', 0)) else 0,
            'Blocks': current.get('A_BLK', 0) if not pd.isna(current.get('A_BLK', 0)) else 0,
            'BlockedShots': current.get('A_BLK_S', 0) if not pd.isna(current.get('A_BLK_S', 0)) else 0
        })
        
        # Opponent stats (for defense) - handle NaN by filling with 0
        self.opponent_stats_average[current['AID']].append({
            'Date': current['Date'],
            'ShotsOnGoal': current.get('H_SOG', 0) if not pd.isna(current.get('H_SOG', 0)) else 0,
            'Goals': current.get('HS', 0) if not pd.isna(current.get('HS', 0)) else 0,
            'Penalties': current.get('H_PEN', 0) if not pd.isna(current.get('H_PEN', 0)) else 0,
            'PowerPlayGoals': current.get('H_PPG', 0) if not pd.isna(current.get('H_PPG', 0)) else 0,
            'Saves': current.get('H_SV', 0) if not pd.isna(current.get('H_SV', 0)) else 0,
            'FaceoffsWon': current.get('H_FO', 0) if not pd.isna(current.get('H_FO', 0)) else 0,
            'Hits': current.get('H_HIT', 0) if not pd.isna(current.get('H_HIT', 0)) else 0,
            'Blocks': current.get('H_BLK', 0) if not pd.isna(current.get('H_BLK', 0)) else 0,
            'BlockedShots': current.get('H_BLK_S', 0) if not pd.isna(current.get('H_BLK_S', 0)) else 0
        })
        
        self.opponent_stats_average[current['HID']].append({
            'Date': current['Date'],
            'ShotsOnGoal': current.get('A_SOG', 0) if not pd.isna(current.get('A_SOG', 0)) else 0,
            'Goals': current.get('AS', 0) if not pd.isna(current.get('AS', 0)) else 0,
            'Penalties': current.get('A_PEN', 0) if not pd.isna(current.get('A_PEN', 0)) else 0,
            'PowerPlayGoals': current.get('A_PPG', 0) if not pd.isna(current.get('A_PPG', 0)) else 0,
            'Saves': current.get('A_SV', 0) if not pd.isna(current.get('A_SV', 0)) else 0,
            'FaceoffsWon': current.get('A_FO', 0) if not pd.isna(current.get('A_FO', 0)) else 0,
            'Hits': current.get('A_HIT', 0) if not pd.isna(current.get('A_HIT', 0)) else 0,
            'Blocks': current.get('A_BLK', 0) if not pd.isna(current.get('A_BLK', 0)) else 0,
            'BlockedShots': current.get('A_BLK_S', 0) if not pd.isna(current.get('A_BLK_S', 0)) else 0
        })

    def _get_team_input_data(self, team_id, date):
        if len(self.team_stats_average[team_id]) <= 5:
            return None
        
        stats = self._get_stats(date, self.team_stats_average[team_id][-100:])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id][-100:])
        
        if stats['Weight'] < 0.01 or opp_stats['Weight'] < 0.01:
            return None
        
        # Hockey-specific metrics with safe division
        # Shooting percentage (goals/shots)
        shooting_pct = stats['Goals'] / max(stats['ShotsOnGoal'], 0.1)
        
        # Save percentage (saves/opponent shots)
        save_pct = stats['Saves'] / max(opp_stats['ShotsOnGoal'], 0.1)
        
        # Power play efficiency (PP goals per opponent penalty)
        pp_efficiency = stats['PowerPlayGoals'] / max(opp_stats['Penalties'], 0.1)
        
        # Penalty kill success rate (1 - opponent PP goals per own penalties)
        pk_success = 1.0 - (opp_stats['PowerPlayGoals'] / max(stats['Penalties'], 0.1))
        
        # Faceoff win percentage
        faceoff_pct = stats['FaceoffsWon'] / max(stats['FaceoffsWon'] + opp_stats['FaceoffsWon'], 0.1)
        
        # Corsi-like metric: shot attempts (SOG + blocked shots)
        corsi_for = stats['ShotsOnGoal'] + stats['BlockedShots']
        corsi_against = opp_stats['ShotsOnGoal'] + opp_stats['BlockedShots']
        corsi_pct = corsi_for / max(corsi_for + corsi_against, 0.1)
        
        # Shot differential per game
        shot_diff = (stats['ShotsOnGoal'] - opp_stats['ShotsOnGoal']) / stats['Weight']
        
        # Goals for/against per game
        goals_for = stats['Goals'] / stats['Weight']
        goals_against = opp_stats['Goals'] / opp_stats['Weight']
        
        return [
            shooting_pct,
            save_pct,
            pp_efficiency,
            pk_success,
            faceoff_pct,
            corsi_pct,
            shot_diff,
            goals_for,
            goals_against
        ]

    def get_input_data(self, home_id, away_id, date):
        val1 = self._get_team_input_data(home_id, date)
        val2 = self._get_team_input_data(away_id, date)
        
        if val1 is None or val2 is None:
            return None
        
        # Check for NaN values
        if any(math.isnan(v) or math.isinf(v) for v in val1 + val2):
            return None
        
        if len(self.past_pred) < self.lr_required_n:
            return None
        
        if self.lr_retrain <= 0:
            self.lr_retrain = 200
            
            # Filter out entries with NaN
            valid_pred = []
            for entry in self.past_pred:
                if not any(math.isnan(v) or math.isinf(v) for v in entry):
                    valid_pred.append(entry)
            
            if len(valid_pred) < self.lr_required_n:
                return None
            
            np_array = np.array(valid_pred)
            sample_weights = np.exp(-0.0003 * np.arange(len(valid_pred)))
            self.lr = LogisticRegression(max_iter=10000)
            self.lr.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
        
        return [
            inverse_sigmoid(self.lr.predict_proba(np.array([val1]))[0, 1]) -
            inverse_sigmoid(self.lr.predict_proba(np.array([val2]))[0, 1])
        ]


class GradientDescent:
    """Gradient descent based team rating system"""
    def __init__(self, num_teams=50, monthly_decay=0.8, long_term_decay=0.97, 
                 home_advantage=0.3, goal_variance=1.5, max_epochs=12, min_games=500):
        """
        Args:
            num_teams: Maximum number of unique teams to track
            monthly_decay: Decay factor for monthly time weighting
            long_term_decay: Decay factor for long-term weighting
            home_advantage: Initial home ice advantage parameter
            goal_variance: Initial goal variance (sigma) parameter
            max_epochs: Maximum training epochs per fit
            min_games: Minimum games before making predictions
        """
        self.num_teams = num_teams
        self.monthly_decay = monthly_decay
        self.long_term_decay = long_term_decay
        self.max_epochs = max_epochs
        self.min_games = min_games
        
        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
        self.team_home_advantages = torch.ones(num_teams) * 0
        self.home_advantage = nn.Parameter(torch.tensor(home_advantage))
        self.sigma = nn.Parameter(torch.tensor(goal_variance))
        
        self.games = torch.zeros((50000, 5), dtype=torch.long)
        self.game_count = 0
        
        self.my_team_id = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season
            self.team_sigmas = torch.ones_like(self.team_sigmas) * 20

    def _get_time_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]
        return (self.monthly_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)

    def _get_long_term_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]
        return (self.long_term_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)

    def forward(self, weights, long_term_weights, idx_start):
        games = self.games[idx_start:self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]
        teams_home_advs = self.team_home_advantages[games[:, 1]]
        
        expectations_home = (
            self.home_advantage * long_term_weights[idx_start:]
            + teams_home_advs * long_term_weights[idx_start:]
            + home_ratings - away_ratings
        )
        realities_home = games[:, 3] - games[:, 4]
        
        game_sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 +
            self.team_sigmas[games[:, 2]] ** 2 +
            self.sigma ** 2
        )
        
        log_value = (-0.5 * ((realities_home - expectations_home) / game_sigmas) ** 2
                     - torch.log(game_sigmas)
                     - 0.5 * torch.log(torch.tensor(2 * np.pi)))
        return torch.sum(log_value * weights[idx_start:])

    def _fit(self, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs
            
        weights = self._get_time_weights()
        long_term_weights = self._get_long_term_weights()
        idx_start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam([self.team_mus, self.sigma, self.home_advantage, self.team_home_advantages], lr=0.02)
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.forward(weights, long_term_weights, idx_start)
            loss.backward()
            optimizer.step()

    def add_game(self, current):
        self._check_season(current['Season'])
        
        timestamp = int(current['Date'].timestamp() * 1000)
        team_home = self._map_team_id(current['HID'])
        team_away = self._map_team_id(current['AID'])
        score_home = current['HS']
        score_away = current['AS']
        
        self.games[self.game_count] = torch.tensor(
            [timestamp, team_home, team_away, score_home, score_away], dtype=torch.long
        )
        self.game_count += 1
        
        game_sigma2 = math.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )
        
        self.team_sigmas[team_home] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_home] ** 2 + 1 / game_sigma2)
        self.team_sigmas[team_away] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_away] ** 2 + 1 / game_sigma2)
        
        self.fit_date = None

    def _map_team_id(self, team_id):
        if team_id not in self.my_team_id:
            self.my_team_id[team_id] = len(self.my_team_id)
        return self.my_team_id[team_id]

    def get_input_data(self, home_id, away_id, season, date):
        self._check_season(season)
        
        if self.game_count < self.min_games:
            return None
        
        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date
            self._fit()
        
        team_home = self._map_team_id(home_id)
        team_away = self._map_team_id(away_id)
        
        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = torch.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )
        
        return [game_exp.item() / game_sigma.item()]


class NateSilverElo:
    """Nate Silver's ELO rating system adapted for hockey"""
    def __init__(self, k_base=15, home_advantage=35, ot_multiplier=0.7, 
                 margin_cap=3, season_carryover=0.75):
        """
        Args:
            k_base: Base K-factor for ELO updates (lower = more stable)
            home_advantage: Home ice advantage in ELO points (~35 for NHL)
            ot_multiplier: Multiplier for OT/SO wins (partial wins)
            margin_cap: Maximum goal differential to consider (ignores blowouts)
            season_carryover: Fraction of ELO to carry over to new season
        """
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.ot_multiplier = ot_multiplier
        self.margin_cap = margin_cap
        self.season_carryover = season_carryover
        
        self.elo_map = defaultdict(float)
        self.last_season = -1

    def _new_season(self):
        for key in self.elo_map:
            self.elo_map[key] *= self.season_carryover

    def _win_probability(self, x):
        return 1 / (1 + (math.exp(-x / 175)))

    def add_game(self, current):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HS']
        away_score = current['AS']
        is_ot_or_so = current.get('Special', '') in ['OT', 'PS', 'SO']
        
        if season > self.last_season:
            self.last_season = season
            self._new_season()
        
        # Home ice advantage ~55% in NHL (lower than basketball)
        home_prediction = self._win_probability(self.elo_map[home_id] + self.home_advantage - self.elo_map[away_id])
        away_prediction = 1 - home_prediction
        
        k_factor = self.get_k_factor(home_score - away_score, self.elo_map[home_id] + self.home_advantage, self.elo_map[away_id])
        
        # Reduce rating change for OT/SO wins (they indicate evenly matched teams)
        ot_so_mult = self.ot_multiplier if is_ot_or_so else 1.0
        
        self.elo_map[home_id] += k_factor * ot_so_mult * (current['H'] - home_prediction)
        self.elo_map[away_id] += k_factor * ot_so_mult * (current['A'] - away_prediction)

    def get_input_data(self, home_id, away_id, season):
        if season > self.last_season:
            self.last_season = season
            self._new_season()
        
        return [self.elo_map[home_id] - self.elo_map[away_id] + self.home_advantage]

    def get_k_factor(self, score_difference, elo_home, elo_away):
        # Lower K-factor for hockey (35-40 vs basketball's 50-60)
        # Cap margin-of-victory at margin_cap goals (garbage time goals don't matter)
        capped_diff = max(-self.margin_cap, min(self.margin_cap, score_difference))
        
        if capped_diff > 0:
            return self.k_base * (abs(capped_diff) + 3) ** 0.8 / (7.5 + 0.006 * (elo_home - elo_away))
        else:
            return self.k_base * (abs(capped_diff) + 3) ** 0.8 / (7.5 + 0.006 * (elo_away - elo_home))


class Pythagorean:
    """Pythagorean expectation adapted for hockey with linear/low exponent"""
    def __init__(self, power=1.0, regularization=20, daily_decay=0.992):
        # Hockey uses lower exponent (1.0-1.8) due to lower scoring variance
        self.power = power
        self.regularization = regularization
        self.daily_decay = daily_decay
        self.team_map = defaultdict(list)

    def add_game(self, current):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HS']
        away_score = current['AS']
        
        self.team_map[home_id].append([date, home_score, away_score])
        self.team_map[away_id].append([date, away_score, home_score])

    def _get_weighted(self, team_id, idx, date):
        return sum([x[idx] * (self.daily_decay ** abs((date - x[0]).days)) for x in self.team_map[team_id][-100:]])

    def get_input_data(self, home_id, away_id, date):
        home_scored = self.regularization + self._get_weighted(home_id, 1, date)
        home_allowed = self.regularization + self._get_weighted(home_id, 2, date)
        away_scored = self.regularization + self._get_weighted(away_id, 1, date)
        away_allowed = self.regularization + self._get_weighted(away_id, 2, date)
        
        return [
            (home_scored ** self.power) / (home_scored ** self.power + home_allowed ** self.power),
            (away_scored ** self.power) / (away_scored ** self.power + away_allowed ** self.power)
        ]


class RollingWindowFeatures:
    """Computes rolling window statistics for teams"""
    def __init__(self, window_short=5, window_med=10, window_long=20):
        """
        Args:
            window_short: Short-term rolling window (e.g., last 5 games)
            window_med: Medium-term rolling window (e.g., last 10 games)
            window_long: Long-term rolling window (e.g., last 20 games)
        """
        self.window_short = window_short
        self.window_med = window_med
        self.window_long = window_long
        self.team_history = defaultdict(list)  # Store recent game stats
        
    def add_game(self, current):
        """Add game result to history"""
        home_id = current['HID']
        away_id = current['AID']
        date = current['Date']
        
        # Extract stats for home team
        home_stats = {
            'date': date,
            'goals_for': current['HS'],
            'goals_against': current['AS'],
            'sog_for': current.get('H_SOG', 0) if not pd.isna(current.get('H_SOG')) else 0,
            'sog_against': current.get('A_SOG', 0) if not pd.isna(current.get('A_SOG')) else 0,
            'ppg': current.get('H_PPG', 0) if not pd.isna(current.get('H_PPG')) else 0,
            'pp_opp': current.get('A_PEN', 0) if not pd.isna(current.get('A_PEN')) else 0,
            'faceoff_wins': current.get('H_FO', 0) if not pd.isna(current.get('H_FO')) else 0,
            'faceoff_total': (current.get('H_FO', 0) if not pd.isna(current.get('H_FO')) else 0) + 
                           (current.get('A_FO', 0) if not pd.isna(current.get('A_FO')) else 0),
            'win': 1 if current['H'] else 0
        }
        
        # Extract stats for away team
        away_stats = {
            'date': date,
            'goals_for': current['AS'],
            'goals_against': current['HS'],
            'sog_for': current.get('A_SOG', 0) if not pd.isna(current.get('A_SOG')) else 0,
            'sog_against': current.get('H_SOG', 0) if not pd.isna(current.get('H_SOG')) else 0,
            'ppg': current.get('A_PPG', 0) if not pd.isna(current.get('A_PPG')) else 0,
            'pp_opp': current.get('H_PEN', 0) if not pd.isna(current.get('H_PEN')) else 0,
            'faceoff_wins': current.get('A_FO', 0) if not pd.isna(current.get('A_FO')) else 0,
            'faceoff_total': (current.get('H_FO', 0) if not pd.isna(current.get('H_FO')) else 0) + 
                           (current.get('A_FO', 0) if not pd.isna(current.get('A_FO')) else 0),
            'win': 1 if current['A'] else 0
        }
        
        self.team_history[home_id].append(home_stats)
        self.team_history[away_id].append(away_stats)
        
        # Keep only last max(window_long) * 1.5 games for efficiency
        max_keep = int(self.window_long * 1.5)
        if len(self.team_history[home_id]) > max_keep:
            self.team_history[home_id] = self.team_history[home_id][-max_keep:]
        if len(self.team_history[away_id]) > max_keep:
            self.team_history[away_id] = self.team_history[away_id][-max_keep:]
    
    def _compute_window_stats(self, team_id, window_size):
        """Compute statistics over a rolling window"""
        if team_id not in self.team_history or len(self.team_history[team_id]) < window_size:
            return None
        
        recent_games = self.team_history[team_id][-window_size:]
        
        # Compute averages
        avg_goals_for = np.mean([g['goals_for'] for g in recent_games])
        avg_goals_against = np.mean([g['goals_against'] for g in recent_games])
        avg_sog_for = np.mean([g['sog_for'] for g in recent_games])
        avg_sog_against = np.mean([g['sog_against'] for g in recent_games])
        
        # Power play efficiency
        total_ppg = sum(g['ppg'] for g in recent_games)
        total_pp_opp = sum(g['pp_opp'] for g in recent_games)
        pp_pct = total_ppg / max(total_pp_opp, 0.1)
        
        # Faceoff win percentage
        total_fo_wins = sum(g['faceoff_wins'] for g in recent_games)
        total_fo_total = sum(g['faceoff_total'] for g in recent_games)
        fo_win_pct = total_fo_wins / max(total_fo_total, 0.1)
        
        # Win percentage
        win_pct = np.mean([g['win'] for g in recent_games])
        
        # Shooting percentage (goals / shots)
        total_goals = sum(g['goals_for'] for g in recent_games)
        total_sog = sum(g['sog_for'] for g in recent_games)
        shooting_pct = total_goals / max(total_sog, 0.1)
        
        return [
            avg_goals_for,
            avg_goals_against,
            avg_sog_for,
            avg_sog_against,
            pp_pct,
            fo_win_pct,
            win_pct,
            shooting_pct
        ]
    
    def get_input_data(self, home_id, away_id, date):
        """Get rolling window features for both teams across all window sizes"""
        features = []
        
        # For each window size, compute features for home and away teams
        for window_size in [self.window_short, self.window_med, self.window_long]:
            home_stats = self._compute_window_stats(home_id, window_size)
            away_stats = self._compute_window_stats(away_id, window_size)
            
            if home_stats is None or away_stats is None:
                return None
            
            # Add differential features (home - away)
            features.extend([
                home_stats[0] - away_stats[0],  # Goal differential
                home_stats[1] - away_stats[1],  # Goals against differential (inverted)
                home_stats[2] - away_stats[2],  # SOG differential
                home_stats[4] - away_stats[4],  # PP% differential
                home_stats[5] - away_stats[5],  # FO% differential
                home_stats[6] - away_stats[6],  # Win% differential
                home_stats[7] - away_stats[7],  # Shooting% differential
            ])
        
        return features


class PoissonGoalModel:
    """Poisson model for goal prediction - hockey goals as independent Poisson processes"""
    def __init__(self, regularization=3.0, daily_decay=0.992):
        self.regularization = regularization
        self.daily_decay = daily_decay
        self.team_scoring_rates = defaultdict(list)
        self.team_defensive_rates = defaultdict(list)

    def add_game(self, current):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HS']
        away_score = current['AS']
        
        # Track scoring rates
        self.team_scoring_rates[home_id].append([date, home_score])
        self.team_scoring_rates[away_id].append([date, away_score])
        
        # Track defensive rates (goals allowed)
        self.team_defensive_rates[home_id].append([date, away_score])
        self.team_defensive_rates[away_id].append([date, home_score])
        
        # Keep last 100 games for efficiency
        if len(self.team_scoring_rates[home_id]) > 100:
            self.team_scoring_rates[home_id] = self.team_scoring_rates[home_id][-100:]
            self.team_defensive_rates[home_id] = self.team_defensive_rates[home_id][-100:]
        if len(self.team_scoring_rates[away_id]) > 100:
            self.team_scoring_rates[away_id] = self.team_scoring_rates[away_id][-100:]
            self.team_defensive_rates[away_id] = self.team_defensive_rates[away_id][-100:]

    def _get_weighted_rate(self, team_id, date, is_offensive=True):
        """Calculate time-weighted goal rate"""
        if team_id not in self.team_scoring_rates:
            return None
        
        data = self.team_scoring_rates[team_id] if is_offensive else self.team_defensive_rates[team_id]
        
        if len(data) < 5:
            return None
        
        weighted_goals = 0.0
        total_weight = 0.0
        
        for game_date, goals in data[-50:]:
            weight = self.daily_decay ** abs((date - game_date).days)
            weighted_goals += goals * weight
            total_weight += weight
        
        if total_weight < 0.01:
            return None
        
        return weighted_goals / total_weight

    def get_input_data(self, home_id, away_id, date):
        """Return expected goal differential based on Poisson model"""
        # Home team offensive rate adjusted by away team defensive rate
        home_off = self._get_weighted_rate(home_id, date, is_offensive=True)
        away_def = self._get_weighted_rate(away_id, date, is_offensive=False)
        
        # Away team offensive rate adjusted by home team defensive rate
        away_off = self._get_weighted_rate(away_id, date, is_offensive=True)
        home_def = self._get_weighted_rate(home_id, date, is_offensive=False)
        
        if any(x is None for x in [home_off, away_def, away_off, home_def]):
            return None
        
        # League average is around 3.0 goals per game in NHL
        league_avg = 3.0
        
        # Expected goals for each team (adjusted by opponent's defense)
        expected_home = (home_off / league_avg) * away_def
        expected_away = (away_off / league_avg) * home_def
        
        # Return expected goal differential
        return [expected_home - expected_away]


class Model:
    """Main ensemble model for ice hockey betting"""
    
    def __init__(self, 
                 # Ensemble parameters (optimized defaults from 3-way multinomial optimization)
                 ensamble_required_n=700,
                 ensamble_retrain_freq=300,
                 # Sub-model parameters (optimized defaults)
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
                 # Rolling window parameters (optimized defaults)
                 rolling_window_short=5,
                 rolling_window_med=10,
                 rolling_window_long=25,
                 # Betting strategy parameters
                 bet_threshold=0.05,
                 bet_size_base=0.03,
                 bet_kelly_fraction=0.25,
                 bet_min_confidence=0.40,
                 bet_max_exposure=0.15,
                 bet_size_max_multiplier=3.0,
                 use_overround_adjustment=True,
                 confidence_scaling=True):
        """
        Initialize the ensemble model with configurable hyperparameters.
        
        Args:
            ensamble_required_n: Minimum predictions before ensemble trains
            ensamble_retrain_freq: How often to retrain the ensemble
            pythagorean_power: Exponent for Pythagorean expectation (1.0 = linear)
            pythagorean_regularization: Regularization for goal totals
            pythagorean_decay: Daily decay factor for Pythagorean
            four_factor_required_n: Min games for Four Factor predictions
            four_factor_decay: Daily decay for Four Factor stats
            gradient_descent_epochs: Training epochs per fit
            gradient_descent_min_games: Min games before GD predictions
            exhaustion_b2b_penalty: Back-to-back penalty multiplier
            exhaustion_road_penalty: Road trip penalty multiplier
            elo_k_base: Base ELO K-factor
            elo_home_advantage: Home ice advantage in ELO points
            elo_margin_cap: Maximum goal differential to consider
            poisson_regularization: Regularization for Poisson model
            poisson_decay: Daily decay for Poisson rates
            rolling_window_short: Short-term rolling window size (e.g., 5 games)
            rolling_window_med: Medium-term rolling window size (e.g., 10 games)
            rolling_window_long: Long-term rolling window size (e.g., 20 games)
            bet_threshold: Minimum edge over implied odds to place bet (e.g., 0.05 = 5% edge required)
            bet_size_base: Base bet size as fraction of bankroll (e.g., 0.03 = 3% of bankroll)
            bet_kelly_fraction: Kelly criterion fraction for bet sizing (e.g., 0.25 = quarter Kelly)
            bet_min_confidence: Minimum predicted probability to consider betting (e.g., 0.40 = 40%)
            bet_max_exposure: Maximum total exposure per game as fraction of bankroll (e.g., 0.15 = 15%)
            bet_size_max_multiplier: Maximum multiplier on base bet size (e.g., 3.0 = max 3x base)
            use_overround_adjustment: Whether to normalize implied probabilities to remove bookmaker margin
            confidence_scaling: Whether to scale bet size based on prediction confidence
        """
        # Initialize sub-models with parameters
        self.model_list = [
            Pythagorean(
                power=pythagorean_power,
                regularization=pythagorean_regularization,
                daily_decay=pythagorean_decay
            ),
            HockeyFourFactor(
                lr_required_n=four_factor_required_n,
                daily_decay=four_factor_decay
            ),
            GradientDescent(
                max_epochs=gradient_descent_epochs,
                min_games=gradient_descent_min_games
            ),
            Exhaustion(
                b2b_penalty=exhaustion_b2b_penalty,
                road_trip_penalty=exhaustion_road_penalty
            ),
            NateSilverElo(
                k_base=elo_k_base,
                home_advantage=elo_home_advantage,
                margin_cap=elo_margin_cap
            ),
            PoissonGoalModel(
                regularization=poisson_regularization,
                daily_decay=poisson_decay
            ),
            RollingWindowFeatures(
                window_short=rolling_window_short,
                window_med=rolling_window_med,
                window_long=rolling_window_long
            )
        ]
        
        self.ensamble_required_n = ensamble_required_n
        self.ensamble_retrain_freq = ensamble_retrain_freq
        self.past_pred = []
        self.ensamble = None
        self.ensamble_retrain = 0
        self.prediction_map = {}
        
        # Betting strategy parameters
        self.bet_threshold = bet_threshold
        self.bet_size_base = bet_size_base
        self.bet_kelly_fraction = bet_kelly_fraction
        self.bet_min_confidence = bet_min_confidence
        self.bet_max_exposure = bet_max_exposure
        self.bet_size_max_multiplier = bet_size_max_multiplier
        self.use_overround_adjustment = use_overround_adjustment
        self.confidence_scaling = confidence_scaling

    def _get_input_features(self, home_id, away_id, season, date):
        """Get input features from all sub-models"""
        input_data = []
        
        for model in self.model_list:
            if isinstance(model, NateSilverElo):
                data = model.get_input_data(home_id, away_id, season)
            elif isinstance(model, (Pythagorean, Exhaustion, HockeyFourFactor, PoissonGoalModel, RollingWindowFeatures)):
                data = model.get_input_data(home_id, away_id, date)
            else:  # GradientDescent
                data = model.get_input_data(home_id, away_id, season, date)
            
            if data is None:
                return None
            
            input_data.extend(data)
        
        return input_data

    def _game_increment(self, idx, current):
        """Process a new game result"""
        season = current['Season']
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        
        # Encode outcome as 3-class: 0=Away, 1=Draw, 2=Home
        if current['H']:
            outcome = 2  # Home win
        elif current['A']:
            outcome = 0  # Away win
        else:  # current['D']
            outcome = 1  # Draw
        
        # Add training data
        input_arr = self._get_input_features(home_id, away_id, season, date)
        if input_arr is not None:
            self.past_pred.append([*input_arr, outcome])
            self.ensamble_retrain -= 1
        
        # Update sub-models
        for model in self.model_list:
            model.add_game(current)

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        """Place bets based on model predictions"""
        
        # Process incremental data
        for idx in inc.index:
            current = inc.loc[idx]
            self._game_increment(idx, current)
        
        # Initialize bets dataframe
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        
        bets = pd.DataFrame(
            data=np.zeros((len(opps), 3)),
            columns=["BetH", "BetA", "BetD"],
            index=opps.index
        )
        
        # Make predictions and place bets
        for i in opps.index:
            current = opps.loc[i]
            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']
            
            if len(self.past_pred) >= self.ensamble_required_n:
                input_arr = self._get_input_features(home_id, away_id, season, date)
                
                if input_arr is not None:
                    # Retrain ensemble if needed
                    if self.ensamble_retrain <= 0:
                        self.ensamble_retrain = self.ensamble_retrain_freq
                        np_array = np.array(self.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                        # Multinomial logistic regression for 3-class prediction
                        self.ensamble = LogisticRegression(
                            max_iter=10000,
                            multi_class='multinomial',
                            solver='lbfgs'
                        )
                        self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
                    
                    # Get 3-way predictions: [P(Away), P(Draw), P(Home)]
                    probs = self.ensamble.predict_proba(np.array([input_arr]))[0]
                    prob_away, prob_draw, prob_home = probs[0], probs[1], probs[2]
                    self.prediction_map[i] = probs
                    
                    # Get betting odds
                    odds_home = current['OddsH']
                    odds_away = current['OddsA']
                    odds_draw = current['OddsD']
                    
                    # Skip if any odds are invalid
                    if any(odds <= 1.0 for odds in [odds_home, odds_away, odds_draw]):
                        continue
                    
                    # Calculate raw implied probabilities
                    implied_home_raw = 1 / odds_home
                    implied_away_raw = 1 / odds_away
                    implied_draw_raw = 1 / odds_draw
                    
                    # Adjust for overround (bookmaker margin) if enabled
                    if self.use_overround_adjustment:
                        # Normalize implied probabilities to sum to 1.0
                        total_implied = implied_home_raw + implied_away_raw + implied_draw_raw
                        implied_home = implied_home_raw / total_implied
                        implied_away = implied_away_raw / total_implied
                        implied_draw = implied_draw_raw / total_implied
                    else:
                        implied_home = implied_home_raw
                        implied_away = implied_away_raw
                        implied_draw = implied_draw_raw
                    
                    # Calculate confidence using entropy (lower entropy = higher confidence)
                    # Entropy: -sum(p * log(p)) for p in probabilities
                    # Normalize to [0, 1] where 1 = max confidence (entropy=0), 0 = min confidence
                    epsilon = 1e-10  # Avoid log(0)
                    entropy = -(prob_home * np.log(prob_home + epsilon) + 
                               prob_away * np.log(prob_away + epsilon) + 
                               prob_draw * np.log(prob_draw + epsilon))
                    max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
                    confidence = 1.0 - (entropy / max_entropy)  # 0 to 1, higher is more confident
                    
                    # Confidence scaling factor (1.0 if disabled, or scaled by confidence if enabled)
                    confidence_multiplier = confidence if self.confidence_scaling else 1.0
                    
                    # Track total exposure for this game
                    total_exposure = 0.0
                    
                    # Evaluate each outcome for betting
                    bet_opportunities = []
                    
                    # Home win
                    if prob_home >= self.bet_min_confidence:
                        edge_home = prob_home - implied_home
                        if edge_home > self.bet_threshold:
                            # Kelly criterion: f = (p * (b - 1) - (1 - p)) / (b - 1)
                            # where b = decimal odds, p = true probability
                            kelly_fraction = (prob_home * (odds_home - 1) - (1 - prob_home)) / (odds_home - 1)
                            kelly_fraction = max(0, kelly_fraction)  # No negative bets
                            
                            # Apply fractional Kelly and confidence scaling
                            bet_fraction = self.bet_kelly_fraction * kelly_fraction * confidence_multiplier
                            
                            # Cap at max multiplier of base bet size
                            bet_fraction = min(bet_fraction, self.bet_size_base * self.bet_size_max_multiplier)
                            
                            bet_opportunities.append(('BetH', bet_fraction, edge_home))
                    
                    # Away win
                    if prob_away >= self.bet_min_confidence:
                        edge_away = prob_away - implied_away
                        if edge_away > self.bet_threshold:
                            kelly_fraction = (prob_away * (odds_away - 1) - (1 - prob_away)) / (odds_away - 1)
                            kelly_fraction = max(0, kelly_fraction)
                            
                            bet_fraction = self.bet_kelly_fraction * kelly_fraction * confidence_multiplier
                            bet_fraction = min(bet_fraction, self.bet_size_base * self.bet_size_max_multiplier)
                            
                            bet_opportunities.append(('BetA', bet_fraction, edge_away))
                    
                    # Draw
                    if prob_draw >= self.bet_min_confidence:
                        edge_draw = prob_draw - implied_draw
                        if edge_draw > self.bet_threshold:
                            kelly_fraction = (prob_draw * (odds_draw - 1) - (1 - prob_draw)) / (odds_draw - 1)
                            kelly_fraction = max(0, kelly_fraction)
                            
                            bet_fraction = self.bet_kelly_fraction * kelly_fraction * confidence_multiplier
                            bet_fraction = min(bet_fraction, self.bet_size_base * self.bet_size_max_multiplier)
                            
                            bet_opportunities.append(('BetD', bet_fraction, edge_draw))
                    
                    # Sort by edge (highest edge first) and place bets respecting exposure limit
                    bet_opportunities.sort(key=lambda x: x[2], reverse=True)
                    
                    for bet_type, bet_fraction, edge in bet_opportunities:
                        # Check if we've exceeded max exposure for this game
                        if total_exposure >= self.bet_max_exposure:
                            break
                        
                        # Calculate bet amount
                        bet_amount = bankroll * bet_fraction
                        
                        # Respect min/max bet constraints
                        bet_amount = max(min_bet, min(max_bet, bet_amount))
                        
                        # Adjust if this would exceed max exposure
                        remaining_exposure = self.bet_max_exposure - total_exposure
                        if bet_amount / bankroll > remaining_exposure:
                            bet_amount = bankroll * remaining_exposure
                            bet_amount = max(min_bet, bet_amount)  # Still respect min bet
                        
                        # Place the bet
                        bets.at[i, bet_type] = bet_amount
                        total_exposure += bet_amount / bankroll
        
        return bets
