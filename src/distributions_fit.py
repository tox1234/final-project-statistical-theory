import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_distributions(df: pd.DataFrame):
    """
    Evaluates the physical distribution constraints of pollutants.
    Applies KS-test for normality and fits a Gamma PDF via MLE.
    """
    # סינון ערכי אפס או שליליים שגורמים לשגיאת MLE
    pm25 = df['PM2_5'][df['PM2_5'] > 0]
    
    print("\n--- Part A: Distributions and Goodness of Fit ---")
    
    mean_val, std_val = pm25.mean(), pm25.std()
    ks_stat, ks_p_value = stats.kstest(pm25, 'norm', args=(mean_val, std_val))
    
    print(f"Sample Size (n)   : {len(pm25)}")
    print(f"Normal Assumption : Mean = {mean_val:.2f}, Std = {std_val:.2f}")
    print(f"KS Statistic      : {ks_stat:.4f}")
    print(f"KS p-value        : {ks_p_value:.4e}")
    if ks_p_value < 0.05:
        print("Conclusion        : Decisive rejection of Normality (H0) for PM2.5.")
        
    shape, loc, scale = stats.gamma.fit(pm25, floc=0)
    print(f"MLE Gamma Fit     : Shape (\u03b1) = {shape:.4f}, Scale (\u03b2) = {scale:.4f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(pm25, stat='density', bins=50, color='#B0C4DE', edgecolor='black', label='Empirical Data (PM2.5)')
    
    x_axis = np.linspace(0, pm25.max() * 1.1, 1000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_val, std_val), 'r--', linewidth=2, label='Normal Fit (H0 - Rejected)')
    plt.plot(x_axis, stats.gamma.pdf(x_axis, shape, loc=0, scale=scale), 'g-', linewidth=2.5, label='Gamma Fit (MLE)')
    
    plt.title("Empirical Distribution vs. Theoretical Density Functions for PM2.5", fontsize=14)
    plt.xlabel("PM2.5 Concentration (\u03bcg/m\u00b3)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribution_analysis_plot.png', dpi=300)
    plt.close()
    print("Visual distribution plot saved as 'distribution_analysis_plot.png'.")
