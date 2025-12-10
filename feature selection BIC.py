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

list_independent_var = list(independent_var.keys())
unique_pairs = list(combinations(list_independent_var,2))

for pair in unique_pairs:
    df_clean[f'{pair[0]}_{pair[1]}_interaction'] = df_clean[f'{pair[0]}'] * df_clean[f'{pair[1]}']
    list_independent_var.append(f'{pair[0]}_{pair[1]}_interaction')

def log_likelihood(y_true, y_pred):
    n = len(y_true)
    residuals = y_true-y_pred
    sigma2 = np.sum(residuals**2)/n
    logL = (-n/2) * np.log(2*np.pi*sigma2) - (1/(2*sigma2)) * np.sum(residuals**2)
    return logL

def calculate_BIC(k,n,logL):
    return k * np.log(n) - 2 * logL


def plot_k_BIC(data_frame, x,y):
    selected_features = []
    features_at_step = []
    remaining_features = x.copy()
    y_values = df_clean[y]
    list_k = []
    list_BIC = []

    while remaining_features:
        dict_BIC = {}
        for feature in remaining_features:
            test_feature = selected_features + [feature]
            x_test = data_frame[test_feature]
            model = LinearRegression()
            model.fit(x_test,y_values)
            y_pred = model.predict(x_test)

            logL = log_likelihood(y_values,y_pred)
            k = len(test_feature)
            n = len(y_values)
            BIC = calculate_BIC(k,n,logL)
            dict_BIC[feature] = BIC
        
        feature_min_BIC = min(dict_BIC, key = dict_BIC.get)
        min_BIC_score = dict_BIC[feature_min_BIC]
        selected_features.append(feature_min_BIC)
        remaining_features.remove(feature_min_BIC)
        list_BIC.append(min_BIC_score)
        list_k.append(len(selected_features))
        features_at_step.append(selected_features.copy())
    
    min_index = np.argmin(list_BIC)
    min_k = list_k[min_index]
    features_minimum = features_at_step[min_index]

    return list_k, list_BIC, min_k, features_minimum

def forward_selection_BIC(data_frame, x,y):
    dict_BIC = {}
    selected_features = []
    remaining_features = x
    current_best_BIC = np.inf
    y_values = df_clean[y]

    while remaining_features:
        for feature in remaining_features:
            test_feature = selected_features + [feature]
            x_test = data_frame[test_feature]
            model = LinearRegression()
            model.fit(x_test,y_values)
            y_pred = model.predict(x_test)

            logL = log_likelihood(y_values,y_pred)
            k = len(test_feature)
            n = len(y_values)
            BIC = calculate_BIC(k,n,logL)
            dict_BIC[feature] = BIC

        feature_min_BIC = min(dict_BIC, key = dict_BIC.get)
        min_BIC_score = dict_BIC[feature_min_BIC]

        if min_BIC_score < current_best_BIC:
            remaining_features.remove(feature_min_BIC)
            selected_features.append(feature_min_BIC)
            current_best_BIC = min_BIC_score 
        else:
            break

    return selected_features

X = list_independent_var
Y_police = 'trstplc'
Y_politicians = 'trstplt'
data_frame = df_clean

k_police, BIC_police, min_k_police, min_features_police= plot_k_BIC(data_frame,X,Y_police)
k_politicians, BIC_politicians, min_k_politicians, min_features_politicians  = plot_k_BIC(data_frame,X,Y_politicians)
features_selected_police = forward_selection_BIC(data_frame,X,Y_police)
features_selected_politicians = forward_selection_BIC(data_frame,X,Y_politicians)

print(f'the selected features 1 for police are {features_selected_police}')
print(f'the selected features 2 for police are {min_features_police}')
print(f'the selected features 1 for politicians are {features_selected_politicians}')
print(f'the selected features 2 for politicians are{min_features_politicians}')

plt.figure()
plt.plot(k_police, BIC_police)
plt.axvline(min_k_police, color = 'r',label = f'min BIC k = {min_k_police}')
plt.xlabel('k')
plt.ylabel('BIC')
plt.title('BIC score for k features outcome trust in police')
plt.legend()

plt.figure()
plt.plot(k_politicians, BIC_politicians)
plt.axvline(min_k_politicians, color = 'r', label = f'min BIC k = {min_k_politicians}')
plt.xlabel('k')
plt.ylabel('BIC')
plt.title('BIC score for k features outcome trust in politicians')
plt.legend()

plt.show()


