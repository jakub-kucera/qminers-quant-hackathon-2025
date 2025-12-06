import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class SimpleElo:
    def __init__(self, k_factor=20, initial_elo=1500):
        self.k_factor = k_factor
        self.ratings = {} # TeamID -> Rating
        self.initial_elo = initial_elo

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, team_a, team_b, actual_score_a):
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        
        expected_a = self.expected_score(ra, rb)
        
        # Simple update
        new_ra = ra + self.k_factor * (actual_score_a - expected_a)
        new_rb = rb + self.k_factor * ((1 - actual_score_a) - (1 - expected_a))
        
        self.ratings[team_a] = new_ra
        self.ratings[team_b] = new_rb
        
        return ra, rb, expected_a

def simulate_elo():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    elo_system = SimpleElo(k_factor=6) # Lower K-factor for NHL (noisy)
    
    predictions = []
    outcomes = []
    
    print("Simulating ELO...")
    history = []
    
    for _, row in df.iterrows():
        hid = row['HID']
        aid = row['AID']
        h_win = 1.0 if row['H'] else 0.0
        
        prob_h = elo_system.expected_score(elo_system.get_rating(hid), elo_system.get_rating(aid))
        
        # Track before update
        predictions.append(prob_h)
        outcomes.append(h_win)
        
        elo_system.update(hid, aid, h_win)
        
        history.append({
            'Date': row['Date'],
            'Home': hid, 'HomeElo': elo_system.get_rating(hid),
            'Away': aid, 'AwayElo': elo_system.get_rating(aid),
            'ProbH': prob_h,
            'Outcome': h_win
        })
        
    # Analysis
    res_df = pd.DataFrame(history)
    mse = ((np.array(predictions) - np.array(outcomes))**2).mean()
    acc = ((np.array(predictions) > 0.5) == np.array(outcomes)).mean()
    
    print(f"ELO Simulation Results:")
    print(f"MSE: {mse:.5f}")
    print(f"Accuracy: {acc:.2%}")
    
    readme = f"""# ELO Simulation Analysis

## Goal
Verify if a simple ELO model has predictive power on this specific dataset (`games.csv`).

## Method
- Simulated a standard ELO system (K=6) over the entire history.
- **Sequential**: Training happens game-by-game.

## Results
- **MSE**: {mse:.5f}
- **Accuracy**: {acc:.2%}

## Conclusion
A pure ELO models yields ~{acc*100:.1f}% accuracy. This serves as a "Zero-Feature" baseline. The Logistic/Neural models must beat this significantly to verify that other features (SOG, etc.) are adding value.
"""
    os.makedirs('src/research', exist_ok=True)
    with open('src/research/ELO_README.md', 'w') as f:
        f.write(readme)
    print("README generated at src/research/ELO_README.md")

if __name__ == "__main__":
    simulate_elo()
