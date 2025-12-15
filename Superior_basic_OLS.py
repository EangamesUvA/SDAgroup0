from cleaning_file import ESSDataCleaner, missing_codes, mapping
import pandas as pd 
import numpy as np
from test_ols import *
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_FILENAME = "data/all_var_set.csv"

# ========================================
# 0. Load and clean data
# ========================================

cleaner = ESSDataCleaner(DATA_FILENAME)
df_clean = cleaner.full_clean()

# ========================================
# 1. Variables
# ========================================

predictors = ['agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
              'nwspol', 'netustm', 'vote', 'gndr']
outcome = ['trstplc', 'trstplt']

continuous_vars = ['agea', 'nwspol', 'netustm', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl']
categorical_vars = ['vote', 'gndr'] 

# ========================================
# 2. Transform skewed predictors
# ========================================

df_clean['nwspol_log'] = np.log1p(df_clean['nwspol'])  # log(1 + x) to reduce skew

# Update predictors for modeling
predictors_mod = ['agea', 'nwspol_log', 'netustm', 'hinctnta', 
                  'edlvenl', 'edlvfenl', 'edlvmenl', 'vote', 'gndr']

# ========================================
# 3. Checking for Linearity 
# ========================================

check_linearity(df_clean, continuous_vars, 'trstplt')

# ========================================
# 4. Checking for Multicollinearity 
# ========================================

vif_table = VIF_check(df_clean, continuous_vars + ['nwspol_log'])
print(vif_table)

# ========================================
# 5. Fitting OLS with robust SE (HC3)
# ========================================

X = df_clean[predictors_mod]
X = sm.add_constant(X)

y_trstplt = df_clean['trstplt']
y_trstplc = df_clean['trstplc']

# Fit models with HC3 robust standard errors
model_trstplt = sm.OLS(y_trstplt, X).fit(cov_type='HC3')
model_trstplc = sm.OLS(y_trstplc, X).fit(cov_type='HC3')

#HC3 becuase we are dealing with social sciences
#The Q-Q plot did not look promising, although we are dealing wiht a large dataset
#Just validates the p-values, better safe than sorry policy 


print(model_trstplt.summary())
print(model_trstplc.summary())

# ========================================
# 6. Residual diagnostics
# ========================================

# Normality
plot_qq(model_trstplt, dep_var_name='trstplt')
plot_qq(model_trstplc, dep_var_name='trstplc')

# Homoscedasticity
plot_resid_vs_fitted(model_trstplt, dep_var_name='trstplt')
plot_resid_vs_fitted(model_trstplc, dep_var_name='trstplc')

# Outliers and Influence
check_outliers_influence(model_trstplt, dep_var_name='trstplt')
check_outliers_influence(model_trstplc, dep_var_name='trstplc')
