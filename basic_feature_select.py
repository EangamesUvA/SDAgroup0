# ========================================
# 1. Import Libraries
# ========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import itertools
from scipy.stats import spearmanr

# ========================================
# 2. Load Data
# ========================================
DATA_FILENAME = "data/all_var_set.csv"
df = pd.read_csv(DATA_FILENAME, quotechar='"')

# ========================================
# 3. Define Missing Values & Variable Labels
# ========================================
missing_codes = {
    'nwspol': [7777, 8888, 9999],
    'netustm': [6666, 7777, 8888, 9999],
    'trstplc': [77, 88, 99],
    'trstplt': [77, 88, 99],
    'vote':[9],
    'gndr': [9],
    'agea': [999],
    'edlvenl': [5555, 6666, 7777, 8888, 9999],
    'hinctnta': [77, 88, 99],
    'edlvfenl': [5555, 7777, 8888, 9999],
    'edlvmenl': [5555, 7777, 8888, 9999],
    'feethngr': [7,8,9],
    'crmvct': [7,8,9],
    'aesfdrk': [7,8,9],
    'polintr': [7,8,9],
    'trplcnt': [7,8,9],
    'trplcmw': [4,6,7,8,9],
    'stflife': [77,88,99],
    'stfeco': [77,88,99],
    'stfgov': [77,88,99],
    'pplfair': [77,88,99],
    'pplhlp': [77,88,99],
    'ppltrst': [77,88,99]
}

# ========================================
# 4. Clean Data
# ========================================
# Replace missing codes with NaN
df.replace(missing_codes, np.nan, inplace=True)

# Define dependent and full predictor list
dependent_vars = ['trstplc', 'trstplt']
predictors_full = [
    'agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
    'nwspol', 'netustm', 'vote', 'gndr', 'polintr',
    'crmvct', 'feethngr', 'pplfair', 'ppltrst', 'pplhlp',
    'stflife', 'stfeco', 'stfgov', 'aesfdrk', 'trplcnt', 'trplcmw'
]

# Drop rows with missing values in all predictors + dependent vars
all_vars_full = predictors_full + dependent_vars
df_clean = df.dropna(subset=all_vars_full).copy()

# Convert numeric columns
df_clean[all_vars_full] = df_clean[all_vars_full].astype(int)

"""
making it work, hopefully 
"""

"""
Making it work pt.2
"""
import random
import statsmodels.api as sm
import itertools
import pandas as pd

# Assume df_clean and predictors_full are defined
X_all_base = df_clean[predictors_full].copy()

# --- Optionally include interactions ---
include_interactions = True
interaction_dict = {}
if include_interactions:
    for a, b in itertools.combinations(X_all_base.columns, 2):
        interaction_dict[f"{a}_x_{b}"] = X_all_base[a] * X_all_base[b]

if interaction_dict:
    X_all = pd.concat([X_all_base, pd.DataFrame(interaction_dict, index=X_all_base.index)], axis=1)
else:
    X_all = X_all_base.copy()

# --- Function to perform iterative OLS selection ---
def stepwise_ols(X_all, y, start_features=6):
    # Initialize with random features
    current_features = random.sample(list(X_all.columns), start_features)
    
    def fit_ols(features, X, y):
        X_ols = sm.add_constant(X[features])
        model = sm.OLS(y, X_ols).fit()
        return model

    improvement = True
    while improvement:
        improvement = False
        model = fit_ols(current_features, X_all, y)
        pvals = model.pvalues.drop('const', errors='ignore')

        # Remove features with p > 0.05
        high_p = pvals[pvals > 0.05]
        if not high_p.empty:
            worst_feature = high_p.idxmax()
            current_features.remove(worst_feature)
            improvement = True
            continue

        current_r2 = model.rsquared

        # Try adding remaining features
        remaining_features = [f for f in X_all.columns if f not in current_features]
        best_r2 = current_r2
        best_feature_to_add = None
        for f in remaining_features:
            trial_features = current_features + [f]
            trial_model = fit_ols(trial_features, X_all, y)
            trial_pvals = trial_model.pvalues.drop('const', errors='ignore')
            trial_r2 = trial_model.rsquared
            if all(trial_pvals <= 0.05) and trial_r2 > best_r2:
                best_r2 = trial_r2
                best_feature_to_add = f

        if best_feature_to_add:
            current_features.append(best_feature_to_add)
            improvement = True

    # Final model
    final_model = fit_ols(current_features, X_all, y)
    return current_features, final_model

# --- Run for trstplc ---
features_plc, model_plc = stepwise_ols(X_all, df_clean['trstplc'])
print("=== trstplc ===")
print("Final selected features:", features_plc)
print(f"R-squared: {model_plc.rsquared:.3f}, Adjusted R-squared: {model_plc.rsquared_adj:.3f}")
print(model_plc.summary())

# --- Run for trstplt ---
features_plt, model_plt = stepwise_ols(X_all, df_clean['trstplt'])
print("\n=== trstplt ===")
print("Final selected features:", features_plt)
print(f"R-squared: {model_plt.rsquared:.3f}, Adjusted R-squared: {model_plt.rsquared_adj:.3f}")
print(model_plt.summary())

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

def fit_l1_l2_selected(X, y, selected_features):
    """
    Fit Lasso (L1) and Ridge (L2) on a predefined set of features.
    """
    X_sel = X[selected_features]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    
    # Lasso (L1)
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_coef = pd.Series(lasso.coef_, index=selected_features)
    lasso_selected = lasso_coef[lasso_coef != 0]
    
    # Ridge (L2)
    ridge = RidgeCV(cv=5)
    ridge.fit(X_scaled, y)
    ridge_coef = pd.Series(ridge.coef_, index=selected_features)
    
    return lasso_selected, ridge_coef

# Example: using previously selected features from stepwise OLS
# For trstplc
selected_features_plc = ['edlvenl_x_edlvfenl', 'gndr_x_feethngr', 'stflife_x_stfgov',
                         'pplhlp_x_trplcnt', 'feethngr_x_pplfair', 'crmvct_x_trplcmw']

lasso_plc, ridge_plc = fit_l1_l2_selected(X_all, df_clean['trstplc'], selected_features_plc)

print("=== trstplc Lasso Selected Features (non-zero coefficients) ===")
print(lasso_plc)
print("\n=== trstplc Ridge Coefficients ===")
print(ridge_plc)

# For trstplt
selected_features_plt = ['edlvmenl', 'netustm_x_pplhlp', 'hinctnta_x_pplfair',
                         'netustm_x_stflife', 'stflife_x_stfgov', 'polintr_x_stflife']

lasso_plt, ridge_plt = fit_l1_l2_selected(X_all, df_clean['trstplt'], selected_features_plt)

print("\n=== trstplt Lasso Selected Features (non-zero coefficients) ===")
print(lasso_plt)
print("\n=== trstplt Ridge Coefficients ===")
print(ridge_plt)
