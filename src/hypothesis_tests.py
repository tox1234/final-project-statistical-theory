import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def test_independent_samples(df: pd.DataFrame):
    """
    Tests spatial demographics effects via Independent sample procedures.
    Utilizes Mann-Whitney U for non-normality constraints and Welch's for CI.
    """
    print("\n--- Part B: Independent Sample Testing (Population Density) ---")
    median_density = df['Population_Density'].median()
    
    high_density = df[df['Population_Density'] > median_density]['PM2_5']
    low_density = df[df['Population_Density'] <= median_density]['PM2_5']
    
    print(f"Mean PM2.5 -> High Density Area: {high_density.mean():.2f}")
    print(f"Mean PM2.5 -> Low Density Area : {low_density.mean():.2f}")
    
    t_stat, t_p = stats.ttest_ind(high_density, low_density, equal_var=False)
    diff = high_density.mean() - low_density.mean()
    se = np.sqrt(high_density.var()/len(high_density) + low_density.var()/len(low_density))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    
    print(f"Welch's t-test p-value   : {t_p:.4e}")
    print(f"95% Confidence Interval  : [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    u_stat, u_p = stats.mannwhitneyu(high_density, low_density, alternative='greater')
    print(f"Mann-Whitney U Statistic : {u_stat:.2f}")
    print(f"Mann-Whitney U p-value   : {u_p:.4e}")

def test_paired_samples(df: pd.DataFrame):
    """
    Simulates longitudinal temporal structures to evaluate the Wilcoxon signed-rank 
    test capability of handling dependent pairs.
    """
    print("\n--- Part C: Paired Temporal Variations (Wilcoxon Signed-Rank) ---")
    np.random.seed(42)
    sample_pm25 = df['PM2_5'].sample(200, random_state=42).values
    
    morning_readings = sample_pm25 * 0.8  
    evening_readings = sample_pm25 * 1.05 + np.random.normal(0, 2, 200) 
    
    print(f"Longitudinal Morning Mean: {morning_readings.mean():.2f}")
    print(f"Longitudinal Evening Mean: {evening_readings.mean():.2f}")
    
    w_stat, p_val_w = stats.wilcoxon(evening_readings, morning_readings, alternative='greater')
    print(f"Wilcoxon W Statistic     : {w_stat}")
    print(f"Wilcoxon p-value         : {p_val_w:.4e}")

def test_categorical_independence(df: pd.DataFrame):
    """
    Assesses Spatial Dependency via Pearson's Chi-Square.
    Applies Family-Wise Error Rate control (Bonferroni penalty).
    """
    print("\n--- Part D: Spatial Dependency & Categorical Independence ---")
    bins = [0, 2, 6, np.inf]
    labels = ['Near', 'Medium', 'Far']
    df['Industry_Binned'] = pd.cut(df['Proximity_to_Industrial_Areas'], bins=bins, labels=labels)
    
    contingency = pd.crosstab(df['Air_Quality'], df['Industry_Binned'])
    print("Cross-Tabulation Matrix:")
    print(contingency)
    
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nPearson \u03c7\u00b2 Statistic   : {chi2:.2f}")
    print(f"Degrees of Freedom     : {dof}")
    print(f"Standard p-value       : {p_val:.4e}")
    
    base_alpha = 0.05
    n_comparisons = 6
    alpha_adj = base_alpha / n_comparisons
    print(f"Bonferroni Adj. Alpha  : {alpha_adj:.4f}")
    if p_val < alpha_adj:
         print("Inference: Null hypothesis of spatial independence is decidedly rejected.")
         
    # Generate Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
    plt.title("Spatial Dependency: Proximity to Industry vs. Air Quality Status")
    plt.xlabel("Proximity to Industry (Binned)")
    plt.ylabel("Air Quality Index Classification")
    plt.tight_layout()
    plt.savefig('spatial_independence_heatmap.png', dpi=300)
    plt.close()
