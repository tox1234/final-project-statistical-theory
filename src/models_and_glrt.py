import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_regression_and_glrt(df: pd.DataFrame):
    """
    Constructs an Ordinary Least Squares (OLS) manifold and applies the 
    Generalized Likelihood Ratio Test (GLRT) to evaluate sub-space constraints (ω).
    """
    print("\n--- Part E: OLS Regression & Generalized Likelihood Ratio Test (GLRT) ---")
    
    full_formula = 'PM2_5 ~ Temperature + Humidity + Temperature:Humidity + Population_Density + Proximity_to_Industrial_Areas'
    full_model = smf.ols(formula=full_formula, data=df).fit()
    
    print("Full OLS Model Results extracted (Summary truncated for clarity):")
    print(f"F-statistic: {full_model.fvalue:.2f} (p = {full_model.f_pvalue:.2e})")
    print(f"R-squared  : {full_model.rsquared:.4f} (Indicates missing exogenous variables)")
    print(f"Condition #: {full_model.condition_number:.2e} (Severe Multicollinearity expected)")
    
    # === הוספת קוד ייצור הגרף (Figure 3) ===
    plt.figure(figsize=(8, 6))
    fitted_vals = full_model.fittedvalues
    residuals = full_model.resid_pearson
    
    sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.5, color='steelblue')
    plt.axhline(0, color='black', linestyle='-')
    plt.title("Residuals Analysis of the OLS Regression")
    plt.xlabel("Fitted Values")
    plt.ylabel("Studentized Residuals")
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=300)
    plt.close()
    print("Residuals plot saved as 'residuals_plot.png'.")
    # ==================================

    restricted_formula = 'PM2_5 ~ Temperature + Humidity + Population_Density + Proximity_to_Industrial_Areas'
    restricted_model = smf.ols(formula=restricted_formula, data=df).fit()
    
    lr_stat = -2 * (restricted_model.llf - full_model.llf)
    df_diff = full_model.df_model - restricted_model.df_model
    glrt_p = chi2.sf(lr_stat, df_diff)
    
    print(f"\nGLRT Λ Statistic       : {lr_stat:.4f}")
    print(f"Degrees of Freedom (Δ) : {df_diff}")
    print(f"GLRT p-value           : {glrt_p:.4f}")
    if glrt_p > 0.05:
        print("Conclusion: The interaction parameter is statistically negligible. H0 (ω space) is accepted.")

def run_logistic_modeling(df: pd.DataFrame):
    """
    Translates continuous distributions to discrete Neyman-Pearson binary bounds 
    focusing on minimizing false negatives (Type II errors) in critical domains.
    """
    print("\n--- Part F: Logistic Classification & Error Typology ---")
    df['Is_Hazardous'] = (df['Air_Quality'] == 'Hazardous').astype(int)
    
    features = ['CO', 'SO2', 'NO2', 'PM2_5', 'PM10']
    X = df[features]
    y = df['Is_Hazardous']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42, stratify=y)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy  : {accuracy:.4f} (Global prediction correctness)")
    print(f"Precision : {precision:.4f} (Validity of hazardous alerts)")
    print(f"Recall    : {recall:.4f} (Detection power: 1 - P(Type II Error))")
    print(f"ROC-AUC   : {roc_auc:.4f} (Warning: Near-perfect score suggests deterministic leakage from AQI formulas)")
