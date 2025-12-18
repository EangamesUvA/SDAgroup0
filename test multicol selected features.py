from test_ols import VIF_check
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
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


DATA_FILENAME = "data/allvariables.csv"
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
    'edlvmenl': [5555, 7777, 8888, 9999],
    'feethngr': [7,8,9,],
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
    'feethngr': 'Feel part of same race or ethnic group as most people in country',
    'crmvct': 'Respondent or household member victim of burglary/assault last 5 years',
    'aesfdrk': 'Feeling of safety of walking alone in local area after dark',
    'polintr': 'How interested in politics ',
    'trplcnt': 'How fair the police in [country] treat women/men',
    'trplcmw': 'Unfairly treated by the police because being a man/woman',
    'stflife': 'How satisfied with life as a whole',
    'stfeco': 'How satisfied with present state of economy in country',
    'stfgov': 'How satisfied with the national government',
    'pplfair': "Most people are fair",
    'ppltrst': "Most people can be trusted",
    'pplhlp': 'Most people are trying to help '
}

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
    'feethngr': 'Feel part of same race or ethnic group as most people in country',
    'crmvct': 'Respondent or household member victim of burglary/assault last 5 years',
    'aesfdrk': 'Feeling of safety of walking alone in local area after dark',
    'polintr': 'How interested in politics ',
    'trplcnt': 'How fair the police in [country] treat women/men',
    'trplcmw': 'Unfairly treated by the police because being a man/woman',
    'stflife': 'How satisfied with life as a whole',
    'stfeco': 'How satisfied with present state of economy in country',
    'stfgov': 'How satisfied with the national government',
    'pplfair': "Most people are fair",
    'ppltrst': "Most people can be trusted",
    'pplhlp': 'Most people are trying to help '
}

dependent_var = {
    "trstplt": 'trust in politicians'
    
}

# Replace missing codes
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaNs
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

print(df_clean.shape)

list_independent_var = list(independent_var.keys())
unique_pairs = list(combinations(list_independent_var,2))

interaction_cols = [
    f"{a}_{b}_interaction"
    for a, b in combinations(list_independent_var, 2)
]

for (a, b), name in zip(combinations(list_independent_var, 2), interaction_cols):
    df_clean[name] = df_clean[a] * df_clean[b]

all_x_cols = list_independent_var + interaction_cols
df_clean[all_x_cols] = df_clean[all_x_cols].astype(float)


vif = VIF_check(df_clean, predictors=['stfgov', 
                                      'ppltrst_pplhlp_interaction',
                                       'agea_polintr_interaction', 
                                       'agea_stfeco_interaction', 
                                       'vote_agea_interaction',
                                       'stfeco_stfgov_interaction',
                                       'vote_edlvenl_interaction'
                                       ])

print(vif)