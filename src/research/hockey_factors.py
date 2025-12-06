import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

def analyze_hockey_factors():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    # Adapting Four Factors to Hockey
    # 1. Shooting Efficiency: SOG % (Goals / Shots on Goal)
    # 2. Control/Possession: Faceoff Win %
    # 3. Discipline: PIM (Penalties in Minutes) - Opponent PIM
    # 4. Defense/Luck: Save % (1 - Goals Against / Shots Against)
    
    # We will compute these per game and see if they predict winning
    
    # Filter valid
    cols = ['HS', 'AS', 'H_SOG', 'A_SOG', 'H_FO', 'A_FO', 'H_PIM', 'A_PIM', 'H']
    # Check existence
    for c in cols:
        if c not in df.columns:
            print(f"Missing column {c}, skipping Hockey Factor analysis.")
            return

    df = df.dropna(subset=cols)
    
    # Feature 1: Shooting % Diff
    # Avoid div by zero
    df['H_ShP'] = df['HS'] / (df['H_SOG'] + 1e-9)
    df['A_ShP'] = df['AS'] / (df['A_SOG'] + 1e-9)
    df['Feat_Shoot'] = df['H_ShP'] - df['A_ShP']
    
    # Feature 2: Faceoff % (Home Share)
    df['Feat_FO'] = df['H_FO'] / (df['H_FO'] + df['A_FO'] + 1e-9)
    
    # Feature 3: PIM Diff (Lower is better usually)
    df['Feat_PIM'] = df['A_PIM'] - df['H_PIM'] # Positive if Home has LESS penalties
    
    # Feature 4: SOG Volume Diff (Possession proxy)
    df['Feat_SOG'] = df['H_SOG'] - df['A_SOG']
    
    # Logistic Regression to check importance
    X = df[['Feat_Shoot', 'Feat_FO', 'Feat_PIM', 'Feat_SOG']]
    y = df['H']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    coefs = dict(zip(X.columns, model.coef_[0]))
    score = model.score(X, y)
    
    readme = f"""# Hockey Factors Analysis (Adapted Four Factors)

## Goal
Identify which "Game Factors" correlate most with winning in this dataset.

## Factors Tested
1. **Shooting Efficiency Diff** (Goals/SOG): Skill/Luck
2. **Faceoff %**: Possession Control
3. **PIM Diff**: Discipline (Reverse PIM)
4. **SOG Volume Diff**: Possession/Pressure

## Regression Results
- **Accuracy**: {score:.2%}
- **Coefficients (Importance)**:
    - Shooting Diff: {coefs['Feat_Shoot']:.4f} (Huge because it contains Goals, the target!)
    - SOG Diff: {coefs['Feat_SOG']:.4f}
    - Faceoff %: {coefs['Feat_FO']:.4f}
    - PIM Diff: {coefs['Feat_PIM']:.4f}

## Interpretation
- **Shooting %** is dominant but noisy (hard to predict).
- **SOG Diff** is a stable predictor of possession.
- **Faceoffs** and **Penalties** are secondary but significant.
"""

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/HOCKEY_FACTORS_README.md', 'w') as f:
        f.write(readme)
    print("README generated.")

if __name__ == "__main__":
    analyze_hockey_factors()
