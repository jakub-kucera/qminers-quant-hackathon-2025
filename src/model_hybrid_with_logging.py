"""
Hybrid Model with Logging

This extends model_hybrid.py to log various metrics and bankroll over time.
Logs are saved to CSV for later visualization and analysis.
"""

import numpy as np
import pandas as pd
import csv
import os

from model_hybrid import Model as BaseHybridModel

class Model(BaseHybridModel):
    """
    Hybrid model with logging capability
    """
    
    def __init__(self, log_file="hybrid_model_log.csv", **kwargs):
        """
        Initialize with logging capability
        
        Args:
            log_file: Path to CSV file for logging
            **kwargs: All other parameters passed to base model
        """
        super().__init__(**kwargs)
        
        self.log_file = log_file
        self.log_initialized = False
        self.game_counter = 0
        
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
            
            # Basic info
            header = ['game_num', 'date', 'bankroll', 'season']
            
            # Betting info - track number of bets and exposure
            header.extend(['n_bets_today', 'total_exposure_today'])
            
            # Model info - ensemble weights (coeffs) if available
            # Note: The ensemble is a LogisticRegression model trained dynamically
            # We don't know the exact feature names of the sub-models easily without introspection, 
            # but we know there are 7 sub-models + season feature potentially.
            # Based on model.py:
            # 1. NateSilverElo
            # 2. Pythagorean
            # 3. Exhaustion (x2?)
            # 4. HockeyFourFactor
            # 5. GradientDescent
            # 6. RollingWindowFeatures
            # 7. PoissonGoalModel
            # Let's just log the raw coefficients by index
            
            # We'll log up to 10 coefficients just to be safe
            for i in range(10):
                header.append(f'ensemble_coef_{i}')
            
            header.append('ensemble_intercept')
                
            writer.writerow(header)
        
        self.log_initialized = True
    
    def _log_status(self, date, bankroll, season, bets_today):
        """Log current status"""
        
        self._initialize_log_file()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Calculate betting stats for today
            n_bets = 0
            exposure = 0.0
            if not bets_today.empty:
                n_bets = (bets_today[['BetH', 'BetD', 'BetA']] > 0).sum().sum()
                exposure = bets_today[['BetH', 'BetD', 'BetA']].sum().sum()
            
            row = [self.game_counter, date.strftime('%Y-%m-%d'), bankroll, season, n_bets, exposure]
            
            # Add ensemble weights if trained
            if hasattr(self, 'ensamble') and hasattr(self.ensamble, 'coef_'):
                # LogisticRegression (multinomial) coef_ shape is (n_classes, n_features)
                # We usually care about the coefficient for the 'Home Win' class or similar.
                # Let's flatter the Home Win coefficients (typically class index 2 in Home/Draw/Away or similar)
                # Actually, predict_proba gives [Away, Draw, Home] usually?
                # Let's just log the first row of coeffs to start (or flatten all)
                # Ideally we want to know how much it weights each submodel.
                
                # Check shape
                coefs = self.ensamble.coef_
                # If multinomial, it might be (3, n_features). Let's just take the first class (Home? Away?) 
                # or mean abs value to see importance.
                # Let's just log the first row for now (e.g. Away win probs)
                
                flat_coefs = coefs[0] if len(coefs.shape) > 1 else coefs
                
                for i in range(10):
                    if i < len(flat_coefs):
                        row.append(flat_coefs[i])
                    else:
                        row.append(0.0)
                        
                # Intercept
                if hasattr(self.ensamble, 'intercept_'):
                     row.append(self.ensamble.intercept_[0])
                else:
                    row.append(0.0)
            else:
                # No model yet
                for _ in range(11): # 10 coefs + 1 intercept
                    row.append(0.0)
            
            writer.writerow(row)
        
        self.game_counter += 1
    
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        """
        Place bets and log status
        """
        # Call parent's place_bets method to get bets
        bets = super().place_bets(summary, opps, inc)
        
        # Get current bankroll
        bankroll = summary.iloc[0]["Bankroll"]
        
        # Log status
        if len(opps) > 0:
            current_date = opps.iloc[0]['Date']
            current_season = opps.iloc[0]['Season']
            self._log_status(current_date, bankroll, current_season, bets)
        
        return bets
