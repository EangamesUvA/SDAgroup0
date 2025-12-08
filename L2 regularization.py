import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


DATA_FILENAME = "data/ESS11e04_0-subset.csv"
df = pd.read_csv(DATA_FILENAME, quotechar='"')

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
    "netustm": 'internet use/day in minutes',
    "trstplc": 'trust in the police',
    "trstplt": 'trust in politicians',
    "vote": 'Voted in the last election',
    "gndr": 'Gender/Sex',
    "agea": 'Age of respondent, calculated',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Households total net income, all sources',
    "edlvfenl": 'Fathers highest level of education, Netherlands',
    "edlvmenl": 'Mothers highest level of education, Netherlands',
}

# Replace missing codes
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaNs
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

numeric_cols = list(mapping.keys())
df_clean[numeric_cols] = df_clean[numeric_cols].astype(int)

independent_var = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netustm": 'internet use/day in minutes',
    "vote": 'Voted in the last election',
    "gndr": 'Gender/Sex',
    "agea": 'Age of respondent, calculated',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Households total net income, all sources',
    "edlvfenl": 'Fathers highest level of education, Netherlands',
    "edlvmenl": 'Mothers highest level of education, Netherlands',
}

dependent_var = {
    "trstplc": 'trust in the police',
    "trstplt": 'trust in politicians',
}

# splits the data into training and testing data
X_var_train, X_var_test, Y_var_train, Y_var_test = train_test_split(df_clean[independent_var.keys()],df_clean[dependent_var.keys()],test_size=0.3, random_state=42)

# scales the data so higher absolute numbers like income don't dominate the cost function
scaler = StandardScaler()
X_var_train_scaled = scaler.fit_transform(X_var_train)
X_var_test_scaled = scaler.transform(X_var_test)

# create multiple values for alpha (lambda) on different scales (log)
alphas = np.logspace(-3,3,7)
dict_alpha = {'alpha': alphas}

# use a grid search to test different alpha's and see which one predicts the "unseen" data best
grid = GridSearchCV(Ridge(), dict_alpha,scoring = 'neg_mean_squared_error', cv = 5, n_jobs=-1)

grid.fit(X_var_train_scaled, Y_var_train)

best_alpha = grid.best_params_['alpha']
print(f' the best alpha is {best_alpha}')

# training the model with the best alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_var_train_scaled, Y_var_train)

# predicted Y based on unseen test X 
Y_pred = ridge_model.predict(X_var_test_scaled)

# evaluating performance of the model
rmse = np.sqrt(mean_squared_error(Y_var_test, Y_pred))
r2 = r2_score(Y_var_test,Y_pred)
print(f'the rmse is {rmse}, the r-squared is {r2}')



