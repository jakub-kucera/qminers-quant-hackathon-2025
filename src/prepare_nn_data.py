import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def prepare_nn_data(csv_path='../data/games.csv', start_season=2009, window_size=5):
    """
    Prepares data for the Neural Network model.
    Encodes Team IDs, generates rolling features, and normalizes inputs.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort by date to ensure correct rolling windows
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. Team ID Encoding
    # Map raw Team IDs to 0..N-1 for embeddings
    unique_teams = pd.concat([df['HID'], df['AID']]).unique()
    team_to_idx = {team: i for i, team in enumerate(sorted(unique_teams))}
    idx_to_team = {i: team for team, i in team_to_idx.items()}
    
    print(f"Found {len(unique_teams)} unique teams.")
    
    # 2. Feature Engineering (Rolling Windows)
    # We'll use a simplified version of the logic from RollingWindowFeatures
    # We want: Goals For, Goals Against, SOG For, SOG Against, PIM, Hit, Blk
    
    team_stats = defaultdict(list)
    processed_games = []
    
    features_list = []
    targets_list = []
    
    print("Generating rolling features...")
    
    for idx, row in df.iterrows():
        season = row['Season']
        
        # Skip if before start_season (data quality check)
        # But we still process stats to keep history current? 
        # Actually safer to just skip processing for NN training samples if < start_season
        # BUT we might need history for the first few games of start_season?
        # For simplicity, we'll start generating valid samples only after we have window_size games
        
        hid, aid = row['HID'], row['AID']
        date = row['Date']
        
        # Get history for both teams
        h_history = team_stats[hid]
        a_history = team_stats[aid]
        
        # Only create a training sample if both have enough history AND we are in valid season
        if season >= start_season and len(h_history) >= window_size and len(a_history) >= window_size:
            
            # --- Feature Vector Construction ---
            
            # Helper to get avg of last N
            def get_rolling_stats(history, N):
                # history is list of dicts: {'GF', 'GA', 'SOG', 'OPP_SOG', ...}
                recent = history[-N:]
                stats = {}
                for key in recent[0].keys():
                    stats[key] = np.mean([g[key] for g in recent])
                return stats

            h_stats = get_rolling_stats(h_history, window_size)
            a_stats = get_rolling_stats(a_history, window_size)
            
            # Create feature row
            # Format: [Home_ID, Away_ID, H_GF_Avg, H_GA_Avg, ..., A_GF_Avg, ...]
            
            # Numerical features (will be normalized later)
            # Home Team Metrics
            num_feats = [
                h_stats['GF'], h_stats['GA'], 
                h_stats['SOG'], h_stats['OPP_SOG'],
                h_stats['PPG'], h_stats['PIM'],
                # Away Team Metrics
                a_stats['GF'], a_stats['GA'],
                a_stats['SOG'], a_stats['OPP_SOG'],
                a_stats['PPG'], a_stats['PIM']
            ]
            
            # Target: Did Home Win? (Binary)
            # H, A, D columns. We treat it as Binary Classification (Win vs Not Win)
            # Or assume no draws? The data has 'D'. 
            # Simple model: Target is Home Win (1.0) or Not (0.0)
            target = 1.0 if row['H'] else 0.0
            
            processed_games.append({
                'Date': date,
                'Season': season,
                'HID_idx': team_to_idx[hid],
                'AID_idx': team_to_idx[aid],
                'Numerical_Features': num_feats,
                'Target': target,
                'OddsH': row['OddsH'],
                'OddsA': row['OddsA']
            })
            
        # Update History AFTER processing the row (to define "pre-game" stats)
        # Stats to track per game: GF (Goals For), GA (Goals Against), SOG, OPP_SOG, PPG, PIM
        
        # Parse Game Outcome Data
        h_score = row['HS']
        a_score = row['AS']
        h_sog = row['H_SOG'] if not pd.isna(row['H_SOG']) else 0
        a_sog = row['A_SOG'] if not pd.isna(row['A_SOG']) else 0
        h_ppg = row['H_PPG'] if not pd.isna(row['H_PPG']) else 0
        a_ppg = row['A_PPG'] if not pd.isna(row['A_PPG']) else 0
        h_pim = row['H_PIM'] if not pd.isna(row['H_PIM']) else 0
        a_pim = row['A_PIM'] if not pd.isna(row['A_PIM']) else 0
        
        # Store for Home Team
        team_stats[hid].append({
            'GF': h_score, 'GA': a_score,
            'SOG': h_sog, 'OPP_SOG': a_sog,
            'PPG': h_ppg, 'PIM': h_pim
        })
        
        # Store for Away Team
        team_stats[aid].append({
            'GF': a_score, 'GA': h_score, # Swapped for Away perspective
            'SOG': a_sog, 'OPP_SOG': h_sog,
            'PPG': a_ppg, 'PIM': a_pim
        })

    # Prepare Tensors
    print(f"Generated {len(processed_games)} processed samples.")
    
    if len(processed_games) == 0:
        raise ValueError("No games passed the filter! Check start_season and window_size.")

    # Convert to arrays
    X_home_id = np.array([g['HID_idx'] for g in processed_games])
    X_away_id = np.array([g['AID_idx'] for g in processed_games])
    X_num = np.array([g['Numerical_Features'] for g in processed_games])
    y = np.array([g['Target'] for g in processed_games])
    
    # Normalize Numerical Features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # Save processed data
    output_data = {
        'X_home': torch.tensor(X_home_id, dtype=torch.long),
        'X_away': torch.tensor(X_away_id, dtype=torch.long),
        'X_num': torch.tensor(X_num_scaled, dtype=torch.float32),
        'y': torch.tensor(y, dtype=torch.float32).unsqueeze(1), # Shape (N, 1)
        'dates': [g['Date'] for g in processed_games],
        'odds': [(g['OddsH'], g['OddsA']) for g in processed_games],
        'team_mapping': idx_to_team,
        'scaler': scaler,
        'feature_names': [
            'H_GF_Avg', 'H_GA_Avg', 'H_SOG_Avg', 'H_OPP_SOG_Avg', 'H_PPG_Avg', 'H_PIM_Avg',
            'A_GF_Avg', 'A_GA_Avg', 'A_SOG_Avg', 'A_OPP_SOG_Avg', 'A_PPG_Avg', 'A_PIM_Avg'
        ]
    }
    
    torch.save(output_data, 'processed_nn_data.pt')
    print("Data saved to processed_nn_data.pt")
    
    return output_data

if __name__ == "__main__":
    prepare_nn_data()
