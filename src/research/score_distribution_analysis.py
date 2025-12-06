import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

def analyze_score_distribution():
    print("Loading data...")
    try:
        df = pd.read_csv('../../data/games.csv')
    except FileNotFoundError:
        # Try relative path if running from different location
        df = pd.read_csv('data/games.csv')

    # Convert scores to Difference
    # In games.csv, scores are usually HS (Home Score) and AS (Away Score)
    # Checking columns...
    if 'HS' in df.columns and 'AS' in df.columns:
        df['Difference'] = df['HS'] - df['AS']
    elif 'HSC' in df.columns and 'ASC' in df.columns: # Fallback to possible other names
        df['Difference'] = df['HSC'] - df['ASC']
    else:
        raise ValueError("Could not find score columns (HS/AS or HSC/ASC)")

    # Plotting
    os.makedirs('research_outputs', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Difference'], bins=30, kde=True) # Reduced bins for cleaner look
    plt.xlabel('Score Difference (Home - Away)')
    plt.title('Distribution of Score Differences in NHL')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    output_plot = 'research_outputs/score_difference_dist.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    # plt.show() # Disabled for headless execution

    # Distribution Fitting
    print("\nFitting distributions...")
    distributions = [stats.norm, stats.logistic, stats.t] # Removed Expon/Gamma as difference includes negatives
    
    results = []
    data = df['Difference'].dropna()
    
    for distribution in distributions:
        try:
            params = distribution.fit(data)
            # KS Test
            ks_stat, p_value = stats.kstest(data, distribution.name, args=params)
            results.append((distribution.name, ks_stat, p_value))
        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")

    results_df = pd.DataFrame(results, columns=['Distribution', 'KS Statistic', 'p-value'])
    print("\nGoodness of Fit Results (Lower KS is better):")
    print(results_df.sort_values(by='KS Statistic'))

    # Generate README content
    readme_content = f"""# Score Difference Analysis

## Goal
Understand the distribution of the margin of victory (Home Score - Away Score) to see if it follows a Normal distribution or something else (e.g., Logistic, T-dist).

## Findings
- **Visual**: The histogram shows the spread of game outcomes.
- **Statistical**:
{results_df.sort_values(by='KS Statistic').to_markdown(index=False)}

## Implication for Modeling
If the distribution is not Normal (e.g., heavier tails), simple Regression models assuming Normality might underestimate the probability of blowouts or close games. A T-distribution is often better for sports scores.
"""
    
    with open('src/research/SCORE_DIST_README.md', 'w') as f:
        f.write(readme_content)
    print("README generated at src/research/SCORE_DIST_README.md")

if __name__ == "__main__":
    analyze_score_distribution()
