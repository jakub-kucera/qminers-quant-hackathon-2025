"""
Simple Rolling Model with Feature Weight Logging

This extends model_simple_rolling.py to log feature weights and bankroll over time.
Logs are saved to CSV for later visualization and analysis.
"""

import numpy as np
import pandas as pd
import csv
import os

from model_simple_rolling import Model as BaseSimpleRollingModel


class Model(BaseSimpleRollingModel):
    """
    Simple rolling model with feature weight logging capability
    """
    
    def __init__(self, log_file="simple_rolling_weights_log.csv", **kwargs):
        """
        Initialize with logging capability
        
        Args:
            log_file: Path to CSV file for logging weights and bankroll
            **kwargs: All other parameters passed to base model
        """
        super().__init__(**kwargs)
        
        self.log_file = log_file
        self.log_initialized = False
        self.game_counter = 0
        
        # Feature names for the 23 features (in order they appear in feature vector)
        self.feature_names = [
            # Short window (5 games) - 7 features
            'short_goal_diff', 'short_ga_diff', 'short_sog_diff',
            'short_pp_diff', 'short_fo_diff', 'short_win_diff', 'short_sh_diff',
            # Medium window (10 games) - 7 features
            'med_goal_diff', 'med_ga_diff', 'med_sog_diff',
            'med_pp_diff', 'med_fo_diff', 'med_win_diff', 'med_sh_diff',
            # Long window (20 games) - 7 features
            'long_goal_diff', 'long_ga_diff', 'long_sog_diff',
            'long_pp_diff', 'long_fo_diff', 'long_win_diff', 'long_sh_diff',
            # ELO difference - 1 feature
            'elo_diff',
            # Head-to-head - 1 feature
            'h2h_result',
        ]
        
    def _initialize_log_file(self):
        """Initialize CSV log file with headers"""
        if self.log_initialized:
            return
            
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header: game_num, date, bankroll, then weights for each feature
            # Binary classification: coefficients for class 1 (home win)
            header = ['game_num', 'date', 'bankroll']
            
            # Add feature weights
            for feature_name in self.feature_names:
                header.append(f'weight_{feature_name}')
            
            # Also log intercept
            header.append('intercept')
            
            # Log model accuracy
            header.append('accuracy')
            header.append('predictions_made')
            
            writer.writerow(header)
        
        self.log_initialized = True
    
    def _log_feature_weights(self, date, bankroll):
        """Log current feature weights and bankroll"""
        if self.model is None:
            return
        
        self._initialize_log_file()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            row = [self.game_counter, date.strftime('%Y-%m-%d'), bankroll]
            
            # Get coefficients (shape: n_features for binary classification)
            # LogisticRegression with binary classification has coef_ shape (1, n_features)
            coef = self.model.coef_[0]  # Get first (and only) row
            
            # Append all feature weights
            for weight in coef:
                row.append(weight)
            
            # Add intercept
            row.append(self.model.intercept_[0])
            
            # Add accuracy metrics
            if self.predictions_made > 0:
                accuracy = self.predictions_correct / self.predictions_made
                row.append(accuracy)
            else:
                row.append(0.0)
            row.append(self.predictions_made)
            
            writer.writerow(row)
        
        self.game_counter += 1
    
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        """
        Place bets and log feature weights
        """
        # # Process incremental data
        # for idx in inc.index:
        #     current = inc.loc[idx]
        #     self._process_game(current)
        
        # # Check if we need to retrain
        # if len(self.training_data) >= self.required_games and self.retrain_counter <= 0:
        #     self._train_model()
        #     self.retrain_counter = self.retrain_freq
        
        # Get current bankroll
        bankroll = summary.iloc[0]["Bankroll"]
        
        # Log weights after processing games (if model is trained)
        if len(opps) > 0 and self.model is not None:
            first_date = opps.iloc[0]['Date']
            self._log_feature_weights(first_date, bankroll)
        
        # Call parent's place_bets method
        return super().place_bets(summary, opps, inc)
