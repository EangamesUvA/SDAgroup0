# ========================================
# 1. Import Libraries
# ========================================
import csv
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
DATA_FILENAME = "data/ESS11e04_0-subset.csv"
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
    'edlvmenl': [5555, 7777, 8888, 9999]
}

mapping = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netustm": 'Internet use/day in minutes',
    "trstplc": 'Trust in the police',
    "trstplt": 'Trust in politicians',
    "vote": 'Voted in the last election',
    "gndr": 'Gender/Sex',
    "agea": 'Age of respondent',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Household net income',
    "edlvfenl": 'Father education level',
    "edlvmenl": 'Mother education level',
}

# ========================================
# 4. Clean Data
# ========================================
# Replace missing codes with NaN
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaN in variables of interest
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

# Convert numeric variables to integer
numeric_cols = list(mapping.keys())
df_clean[numeric_cols] = df_clean[numeric_cols].astype(int)

# ========================================
# 5. Initial OLS Regression: Trust in Police
# ========================================
predictors = ['agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
              'nwspol', 'netustm', 'vote', 'gndr']
outcome = 'trstplc'

X = sm.add_constant(df_clean[predictors])
y = df_clean[outcome]

model = sm.OLS(y, X).fit()
# print(model.summary())

# ========================================
# 6. Residual Diagnostics
# ========================================
residuals = model.resid

# Residuals vs predictors
for col in predictors:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_clean[col], y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs {col}')
    plt.xlabel(col)
    plt.ylabel('Residuals')
    plt.show()

# ========================================
# 7. Transform skewed predictors (nwspol)
# ========================================
df_clean['nwspol_log'] = np.log1p(df_clean['nwspol'])

# Scatter plot after log transform
plt.figure(figsize=(6,4))
sns.scatterplot(x=df_clean['nwspol_log'], y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs nwspol_log')
plt.xlabel('log1p(nwspol)')
plt.ylabel('Residuals')
plt.show()

# Refit OLS with log-transformed nwspol
predictors_log = ['agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
                  'nwspol_log', 'netustm', 'vote', 'gndr']

X_log = sm.add_constant(df_clean[predictors_log])
model_log = sm.OLS(y, X_log).fit()
print(model_log.summary())

# ========================================
# 8. Multicollinearity Check (VIF)
# ========================================
"""
Checks for Multicollinearity, as part of the OLS assumptions 
"""
vif_data = pd.DataFrame()
vif_data['feature'] = X_log.columns
vif_data['VIF'] = [variance_inflation_factor(X_log.values, i) for i in range(X_log.shape[1])]
print(vif_data)

# ========================================
# 9. Heteroscedasticity Check & Robust Model
# ========================================
"""
Issues with Heteroscedasticity, using robust standard errors. Residual distribution -> non-normal. Using SE' to validate inference.

(I am not a residucal expert, so this needs to be checked out what happened here, but it is part of assumption checking)

Protecitng against non-constant variance (?)

HC3 is used here -> again not an expert, if someone is more knowledgable about this, that would be great
"""
# Residuals vs fitted
fitted = model_log.fittedvalues
plt.figure(figsize=(6,4))
sns.scatterplot(x=fitted, y=model_log.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Breusch-Pagan test
bp_test = het_breuschpagan(model_log.resid, X_log)
labels = ['LM stat', 'LM p-value', 'F-stat', 'F p-value']
print(dict(zip(labels, bp_test)))

# Robust standard errors
robust_model = model_log.get_robustcov_results(cov_type='HC3')
print(robust_model.summary())

# Q-Q plot
sm.qqplot(robust_model.resid, line='45')
plt.title('Q-Q Plot of Robust Residuals')
plt.show()

# Shapiro-Wilk test
stat, p = shapiro(robust_model.resid)
print(f'Statistic={stat:.3f}, p-value={p:.3e}')

# ========================================
# 10. Interaction Terms
# ========================================
"""
Adding moderating effects for age and news, age and internet, education and father's education, education and mother's education 
"""
df_clean['age_news'] = df_clean['agea'] * df_clean['nwspol_log']
df_clean['age_internet'] = df_clean['agea'] * df_clean['netustm']
df_clean['edu_father'] = df_clean['edlvenl'] * df_clean['edlvfenl']
df_clean['edu_mother'] = df_clean['edlvenl'] * df_clean['edlvmenl']

predictors_interactions = predictors_log + ['age_news', 'age_internet', 'edu_father', 'edu_mother']

X_interactions = sm.add_constant(df_clean[predictors_interactions])
model_interactions = sm.OLS(y, X_interactions).fit(cov_type='HC3')
print(model_interactions.summary())

# ========================================
# 11. Trust in Politicians Model
# ========================================
y_trstplt = df_clean['trstplt']
X_trstplt = sm.add_constant(df_clean[predictors_interactions])
model_trstplt = sm.OLS(y_trstplt, X_trstplt).fit(cov_type='HC3')
print(model_trstplt.summary())

# ========================================
# 12. Trust in Police Model
# ========================================
y_trstplt = df_clean['trstplc']
X_trstplt = sm.add_constant(df_clean[predictors_interactions])
model_trstplt = sm.OLS(y_trstplt, X_trstplt).fit(cov_type='HC3')
print(model_trstplt.summary())