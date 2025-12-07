import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

MONTH_MS = 30 * 24 * 60 * 60 * 1000


class MarginRatings:
    """Team strengths fitted to goal margins by gradient descent.

    The home margin is modelled as Normal(home advantage + mu_home - mu_away, sigma), with each
    team's uncertainty shrinking as it plays and reset at the start of a season. The log
    likelihood of past games, weighted by recency, is maximised with a few Adam steps
    whenever the date changes. The feature is the expected margin over its standard deviation.
    """

    def __init__(
        self,
        num_teams=50,
        monthly_decay=0.8,
        long_term_decay=0.97,
        home_advantage=0.3,
        goal_variance=1.5,
        max_epochs=12,
        min_games=500,
    ):
        self.monthly_decay = monthly_decay
        self.long_term_decay = long_term_decay
        self.max_epochs = max_epochs
        self.min_games = min_games
        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
        self.team_home_advantages = torch.ones(num_teams) * 0
        self.home_advantage = nn.Parameter(torch.tensor(home_advantage))
        self.sigma = nn.Parameter(torch.tensor(goal_variance))
        # (timestamp in ms, home index, away index, home goals, away goals)
        self.games = torch.zeros((50000, 5), dtype=torch.long)
        self.game_count = 0
        self.team_index = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season
            self.team_sigmas = torch.ones_like(self.team_sigmas) * 20

    def _recency(self, decay):
        timestamps = self.games[: self.game_count, 0].to(torch.float32)
        last = timestamps[-1]
        return (decay ** (torch.abs(timestamps - last) / MONTH_MS)).to(timestamps.device)

    def _log_likelihood(self, weights, long_term_weights, start):
        games = self.games[start : self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]
        team_home_advantages = self.team_home_advantages[games[:, 1]]
        expected = (
            self.home_advantage * long_term_weights[start:]
            + team_home_advantages * long_term_weights[start:]
            + home_ratings
            - away_ratings
        )
        actual = games[:, 3] - games[:, 4]
        sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 + self.team_sigmas[games[:, 2]] ** 2 + self.sigma**2
        )
        log_value = (
            -0.5 * ((actual - expected) / sigmas) ** 2
            - torch.log(sigmas)
            - 0.5 * torch.log(torch.tensor(2 * np.pi))
        )
        return torch.sum(log_value * weights[start:])

    def _fit(self):
        weights = self._recency(self.monthly_decay)
        long_term_weights = self._recency(self.long_term_decay)
        start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam(
            [self.team_mus, self.sigma, self.home_advantage, self.team_home_advantages], lr=0.02
        )
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            loss = -self._log_likelihood(weights, long_term_weights, start)
            loss.backward()
            optimizer.step()

    def _index(self, team_id):
        if team_id not in self.team_index:
            self.team_index[team_id] = len(self.team_index)
        return self.team_index[team_id]

    def add_game(self, game):
        self._check_season(game["Season"])
        home = self._index(game["HID"])
        away = self._index(game["AID"])
        self.games[self.game_count] = torch.tensor(
            [int(game["Date"].timestamp() * 1000), home, away, game["HS"], game["AS"]],
            dtype=torch.long,
        )
        self.game_count += 1
        game_sigma2 = math.sqrt(
            self.team_sigmas[home] ** 2 + self.team_sigmas[away] ** 2 + self.sigma**2
        )
        self.team_sigmas[home] = 1 / torch.sqrt(1 / self.team_sigmas[home] ** 2 + 1 / game_sigma2)
        self.team_sigmas[away] = 1 / torch.sqrt(1 / self.team_sigmas[away] ** 2 + 1 / game_sigma2)
        self.fit_date = None

    def features(self, home_id, away_id, season, date):
        self._check_season(season)
        if self.game_count < self.min_games:
            return None
        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date
            self._fit()
        home = self._index(home_id)
        away = self._index(away_id)
        expected = self.home_advantage + self.team_mus[home] - self.team_mus[away]
        sigma = torch.sqrt(
            self.team_sigmas[home] ** 2 + self.team_sigmas[away] ** 2 + self.sigma**2
        )
        return [expected.item() / sigma.item()]
