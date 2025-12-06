import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import datetime

# ==========================================
# 1. Data Preparation Logic
# ==========================================

def prepare_data(csv_path='../data/games.csv', start_season=2009, window_size=5):
    """
    Reads games.csv, generates rolling features, encodes teams, and normalizes.
    Returns:
        dict containing tensors and metadata
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None

    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Encode Team IDs
    unique_teams = pd.concat([df['HID'], df['AID']]).unique()
    team_to_idx = {team: i for i, team in enumerate(sorted(unique_teams))}
    idx_to_team = {i: team for team, i in team_to_idx.items()}
    num_teams = len(unique_teams)
    
    print(f"Found {num_teams} unique teams.")
    
    team_stats = defaultdict(list)
    processed_games = []
    
    print("Generating rolling features...")
    
    for idx, row in df.iterrows():
        season = row['Season']
        hid, aid = row['HID'], row['AID']
        date = row['Date']
        
        h_history = team_stats[hid]
        a_history = team_stats[aid]
        
        # Valid sample check
        if season >= start_season and len(h_history) >= window_size and len(a_history) >= window_size:
            
            # Feature extraction
            def get_rolling_avg(history, N):
                recent = history[-N:]
                stats = {}
                for key in recent[0].keys():
                    stats[key] = np.mean([g[key] for g in recent])
                return stats

            h_stats = get_rolling_avg(h_history, window_size)
            a_stats = get_rolling_avg(a_history, window_size)
            
            # --- Numerical Feature Vector ---
            # 12 Features currently
            num_feats = [
                h_stats['GF'], h_stats['GA'], 
                h_stats['SOG'], h_stats['OPP_SOG'],
                h_stats['PPG'], h_stats['PIM'],
                # Away data
                a_stats['GF'], a_stats['GA'],
                a_stats['SOG'], a_stats['OPP_SOG'],
                a_stats['PPG'], a_stats['PIM']
            ]
            
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
            
        # Update History
        h_score = row['HS']
        a_score = row['AS']
        # Handle NaNs safely
        def safe_get(val): return val if not pd.isna(val) else 0.0
        
        h_sog = safe_get(row['H_SOG'])
        a_sog = safe_get(row['A_SOG'])
        h_ppg = safe_get(row['H_PPG'])
        a_ppg = safe_get(row['A_PPG'])
        h_pim = safe_get(row['H_PIM'])
        a_pim = safe_get(row['A_PIM'])
        
        team_stats[hid].append({
            'GF': h_score, 'GA': a_score, 'SOG': h_sog, 'OPP_SOG': a_sog, 'PPG': h_ppg, 'PIM': h_pim
        })
        team_stats[aid].append({
            'GF': a_score, 'GA': h_score, 'SOG': a_sog, 'OPP_SOG': h_sog, 'PPG': a_ppg, 'PIM': a_pim
        })

    print(f"Generated {len(processed_games)} processed samples.")
    
    if not processed_games:
        raise ValueError("No data generated with current settings.")

    # Convert to Tensors
    X_home = np.array([g['HID_idx'] for g in processed_games])
    X_away = np.array([g['AID_idx'] for g in processed_games])
    X_num = np.array([g['Numerical_Features'] for g in processed_games])
    y = np.array([g['Target'] for g in processed_games])
    
    # Scale numerical inputs
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    return {
        'X_home': torch.tensor(X_home, dtype=torch.long),
        'X_away': torch.tensor(X_away, dtype=torch.long),
        'X_num': torch.tensor(X_num_scaled, dtype=torch.float32),
        'y': torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        'dates': [g['Date'] for g in processed_games],
        'odds': [(g['OddsH'], g['OddsA']) for g in processed_games],
        'num_teams': num_teams,
        'num_features': X_num.shape[1],
        'team_mapping': idx_to_team
    }

# ==========================================
# 2. PyTorch Dataset & Model
# ==========================================

class NHLDataset(Dataset):
    def __init__(self, data_dict):
        self.X_home = data_dict['X_home']
        self.X_away = data_dict['X_away']
        self.X_num = data_dict['X_num']
        self.y = data_dict['y']
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_home[idx], self.X_away[idx], self.X_num[idx], self.y[idx]

class NHL_MLP(nn.Module):
    def __init__(self, num_teams, num_numerical, embedding_dim=8, hidden_units=[64, 32]):
        super(NHL_MLP, self).__init__()
        
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Input dim = 2 * embedding_dim (Home+Away) + num_numerical
        input_dim = (2 * embedding_dim) + num_numerical
        
        # Dense Layers
        layers = []
        in_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # Prevent overfitting
            in_dim = units
            
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, home_id, away_id, num_features):
        # Lookup embeddings
        h_emb = self.team_embedding(home_id)
        a_emb = self.team_embedding(away_id)
        
        # Concatenate: [Home_Emb, Away_Emb, Numerical_Stats]
        combined = torch.cat([h_emb, a_emb, num_features], dim=1)
        
        return self.network(combined)

# ==========================================
# 3. Training Loop
# ==========================================

def train_nn_pipeline():
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Metal) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # 1. Prepare Data
    data = prepare_data()
    if data is None: return

    # 2. Time-Based Split (Last 20% for validation/testing)
    total_samples = len(data['y'])
    train_size = int(total_samples * 0.8)
    
    # Slicing datasets (Simplistic approach, for DataLoader we need to subset)
    # Actually easier to slice the tensors first
    train_data = {k: v[:train_size] for k, v in data.items() if isinstance(v, torch.Tensor)}
    val_data = {k: v[train_size:] for k, v in data.items() if isinstance(v, torch.Tensor)}
    
    train_dataset = NHLDataset(train_data)
    val_dataset = NHLDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Shuffle train only
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. Initialize Model
    model = NHL_MLP(
        num_teams=data['num_teams'], 
        num_numerical=data['num_features'],
        embedding_dim=16
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("-" * 30)
    print("Starting Training...")
    print("-" * 30)
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for h_id, a_id, num, y_batch in train_loader:
            h_id, a_id, num, y_batch = h_id.to(device), a_id.to(device), num.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(h_id, a_id, num)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for h_id, a_id, num, y_batch in val_loader:
                h_id, a_id, num, y_batch = h_id.to(device), a_id.to(device), num.to(device), y_batch.to(device)
                
                preds = model(h_id, a_id, num)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                # Accuracy
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")

    # 4. Final Evaluation (Profitability Check - Simplified)
    print("\nEvaluating Profitability on Validation Set...")
    # Get raw validation arrays
    val_odds = data['odds'][train_size:]
    
    bankroll = 1000
    bets_made = 0
    wins = 0
    
    model.eval()
    with torch.no_grad():
        # Process full batch for speed
        h_id = val_data['X_home'].to(device)
        a_id = val_data['X_away'].to(device)
        num = val_data['X_num'].to(device)
        y_true = val_data['y'].to(device)
        
        preds = model(h_id, a_id, num).cpu().numpy()
        y_true = y_true.cpu().numpy()
        
    for i in range(len(preds)):
        prob_home = preds[i][0]
        outcome = y_true[i][0] # 1.0 or 0.0
        odds_h,_ = val_odds[i]
        
        # Simple Strategy: Bet if Edge > 5%
        # Implied prob = 1/odds (Normalized odds logic omitted for brevity, using raw)
        if odds_h > 1.0:
            implied_prob = 1.0 / odds_h
            edge = prob_home - implied_prob
            
            if edge > 0.05: # 5% edge
                bets_made += 1
                stake = 50 # Flat bet
                
                if outcome == 1.0:
                    profit = stake * (odds_h - 1)
                    bankroll += profit
                    wins += 1
                else:
                    bankroll -= stake
                    
    print(f"Final Bankroll: ${bankroll:.2f} (Start: $1000)")
    print(f"Bets Made: {bets_made} | Wins: {wins} | Win Rate: {wins/bets_made if bets_made>0 else 0:.1%}")

if __name__ == "__main__":
    train_nn_pipeline()
