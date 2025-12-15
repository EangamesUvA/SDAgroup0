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

# ========================================
# 5. Transform skewed predictors
# ========================================
df_clean['nwspol_log'] = np.log1p(df_clean['nwspol'])

# Create interaction terms
df_clean['age_news'] = df_clean['agea'] * df_clean['nwspol_log']
df_clean['age_internet'] = df_clean['agea'] * df_clean['netustm']
df_clean['edu_father'] = df_clean['edlvenl'] * df_clean['edlvfenl']
df_clean['edu_mother'] = df_clean['edlvenl'] * df_clean['edlvmenl']

# Updated predictor list including transformations and interactions
predictors_final = [
    'agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
    'nwspol_log', 'netustm', 'vote', 'gndr', 'polintr',
    'crmvct', 'feethngr', 'pplfair', 'ppltrst', 'pplhlp',
    'stflife', 'stfeco', 'stfgov', 'aesfdrk', 'trplcnt', 'trplcmw',
    'age_news', 'age_internet', 'edu_father', 'edu_mother'
]

# ========================================
# 6. OLS Regression Function
# ========================================
def run_ols(y_var, X_vars, df):
    X = sm.add_constant(df[X_vars])
    y = df[y_var]
    
    # Fit robust model
    model = sm.OLS(y, X).fit(cov_type='HC3')
    print(f"\n===== OLS Results for {y_var} =====")
    print(model.summary())
    
    # VIF
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVIF:\n", vif_data)
    
    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, X)
    labels = ['LM stat', 'LM p-value', 'F-stat', 'F p-value']
    print("\nBreusch-Pagan test:", dict(zip(labels, bp_test)))
    
    # Shapiro-Wilk for residual normality
    stat, p = shapiro(model.resid)
    print(f"\nShapiro-Wilk test: Statistic={stat:.3f}, p-value={p:.3e}")
    
    # Residuals vs fitted plot
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted for {y_var}')
    plt.show()
    
    # Q-Q plot
    sm.qqplot(model.resid, line='45')
    plt.title(f'Q-Q Plot of Residuals for {y_var}')
    plt.show()
    
    return model

# ========================================
# 7. Run models for both dependent variables
# ========================================
model_trstplc = run_ols('trstplc', predictors_final, df_clean)
model_trstplt = run_ols('trstplt', predictors_final, df_clean)
