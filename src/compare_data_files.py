import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

FILES = [
    os.path.join(DATA_DIR, "games.csv"),
    os.path.join(DATA_DIR, "train_games.csv"),
    os.path.join(DATA_DIR, "bookie_rookie_games.csv")
]

def analyze_file(filepath):
    print(f"\n{'='*50}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print("File not found.")
        return None

    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=['Date', 'Open'])
        
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Seasons: {sorted(df['Season'].unique())}")
        
        # Check for NaN values in critical columns
        missing_scores = df[['HS', 'AS']].isna().sum().sum()
        print(f"Missing Scores (HS/AS): {missing_scores}")
        
        return df
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def analyze_margins(name, df):
    odds_cols = ['OddsH', 'OddsA', 'OddsD']
    if not all(c in df.columns for c in odds_cols):
        return None
    
    valid_mask = (df['OddsH'] > 1) & (df['OddsA'] > 1) & (df['OddsD'] > 1)
    valid_df = df.loc[valid_mask]
    
    if valid_df.empty:
        return None
        
    margins = (1/valid_df['OddsH'] + 1/valid_df['OddsA'] + 1/valid_df['OddsD']) - 1
    return margins

def compare_dataframes(dfs):
    print(f"\n{'='*50}")
    print(f"MARGIN ANALYSIS")
    print(f"{'='*50}")
    
    # Analyze margins independently
    for name, df in dfs.items():
        margins = analyze_margins(name, df)
    
    # Pre-process DFs to use a composite key for alignment
    aligned_dfs = {}
    for name, df in dfs.items():
        # Create a copy to avoid modifying original
        temp_df = df.copy()
        
        # Ensure required columns exist
        req_cols = ['Date', 'HID', 'AID']
        if all(c in temp_df.columns for c in req_cols):
            # Set multi-index for alignment
            temp_df['HID'] = temp_df['HID'].astype(int)
            temp_df['AID'] = temp_df['AID'].astype(int)
            temp_df = temp_df.set_index(req_cols)
            # Filter duplicates if any
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            aligned_dfs[name] = temp_df
        else:
            aligned_dfs[name] = df

    # Compare games.csv vs others AND train_games.csv vs bookie_rookie_games.csv
    comparisons = [
        ("games.csv", "train_games.csv"),
        ("games.csv", "bookie_rookie_games.csv"),
        ("train_games.csv", "bookie_rookie_games.csv")
    ]
    
    for name1, name2 in comparisons:
        if name1 not in aligned_dfs or name2 not in aligned_dfs:
            continue
            
        df1 = aligned_dfs[name1]
        df2 = aligned_dfs[name2]
        
        print(f"\nComparing {name1} vs {name2}:")
        
        common = df1.index.intersection(df2.index)
        print(f"  Overlapping games: {len(common)}")
        
        if len(common) == 0:
            continue
            
        # Check odds differences
        odds_cols = ['OddsH', 'OddsA', 'OddsD']
        diff_count = 0
        for col in odds_cols:
            if col in df1.columns and col in df2.columns:
                s1 = df1.loc[common][col]
                s2 = df2.loc[common][col]
                diffs = (abs(s1 - s2) > 0.0001) & ~(s1.isna() & s2.isna())
                if diffs.any():
                    diff_count += 1
        
        if diff_count > 0:
            print(f"  ODDS DIFFER on {diff_count}/3 columns.")
            
            # Margin comparison on overlap
            m1 = analyze_margins(name1, df1.loc[common])
            m2 = analyze_margins(name2, df2.loc[common])
            
            if m1 is not None and m2 is not None:
                # Re-align in case validity differs
                valid_idx = m1.index.intersection(m2.index)
                m1 = m1.loc[valid_idx]
                m2 = m2.loc[valid_idx]
                
                diff = m2 - m1
                print(f"  Mean Margin Difference ({name2} - {name1}): {diff.mean():.4%}")
                
                lower = (diff < -0.0001).sum()
                higher = (diff > 0.0001).sum()
                print(f"  {name2} lower margin (better odds): {lower} games")
                print(f"  {name2} higher margin (worse odds): {higher} games")
                
                if lower > higher:
                    print(f"  -> {name2} has BETTER odds.")
                elif higher > lower:
                    print(f"  -> {name1} has BETTER odds.")
                else:
                    print(f"  -> Odds quality is mixed.")
        else:
            print("  Odds match EXACTLY in overlapping games.")

def main():
    dfs = {}
    for f in FILES:
        df = analyze_file(f)
        if df is not None:
            dfs[os.path.basename(f)] = df
            
    if len(dfs) > 1:
        compare_dataframes(dfs)

if __name__ == "__main__":
    main()
