import pandas as pd
import networkx as nx
import os

def analyze_clusters():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except:
        df = pd.read_csv('data/games.csv')
        
    # Use NetworkX to find clusters (Conferences) based on who plays whom
    # Teams in the same conference play each other more often usually
    
    G = nx.Graph()
    
    # Edge weight = Number of games
    pair_counts = df.groupby(['HID', 'AID']).size().reset_index(name='count')
    
    for _, row in pair_counts.iterrows():
        G.add_edge(row['HID'], row['AID'], weight=row['count'])
        
    # Detect communities (Louvain or Greedy Modularity)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        print(f"Detected {len(communities)} Communities (Conferences/Divisions)")
        
        readme_content = f"""# Team Clustering (Conference Detection)

## Goal
Detect inherent structure (Conferences) from the schedule graph.

## Results
- **Communities Found**: {len(communities)}
"""
        for i, c in enumerate(communities):
            readme_content += f"\n### Cluster {i+1} ({len(c)} teams)\nTeams: {sorted(list(c))}\n"
            
    except ImportError:
        readme_content = "Could not import community detection algorithms."
        print("Community detection module missing.")

    os.makedirs('src/research', exist_ok=True)
    with open('src/research/CLUSTERING_README.md', 'w') as f:
        f.write(readme_content)
    print("README generated.")

if __name__ == "__main__":
    analyze_clusters()
