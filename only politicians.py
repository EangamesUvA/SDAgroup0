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


list_independent_var = list(independent_var.keys())



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

X = list_independent_var
Y_politicians = 'trstplt'
data_frame = df_clean

k_politicians, BIC_politicians, min_k_politicians, min_features_politicians  = plot_k_BIC(data_frame,X,Y_politicians)


print(f'the selected features for politicians are{min_features_politicians}')

plt.figure()
plt.plot(k_politicians, BIC_politicians)
plt.axvline(min_k_politicians, color = 'r', label = f'min BIC k = {min_k_politicians}')
plt.xlabel('k')
plt.ylabel('BIC')
plt.title('BIC score for k features outcome trust in politicians')
plt.legend()

all_x_cols = min_features_politicians

# splits the data into training and testing data
X_var_train, X_var_test, Y_var_train, Y_var_test = train_test_split(df_clean[all_x_cols],df_clean[dependent_var.keys()],test_size=0.3, random_state=42)

# scales the data so higher absolute numbers like income don't dominate the cost function
#scaler = StandardScaler()
#X_var_train_scaled = scaler.fit_transform(X_var_train)
#X_var_test_scaled = scaler.transform(X_var_test)

#--------------------------------------
# create interaction effects
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_var_train_inter = poly.fit_transform(X_var_train)
X_var_test_inter = poly.transform(X_var_test)


# scales the data so higher absolute numbers like income don't dominate the cost function
scaler = StandardScaler()
X_var_train_scaled = scaler.fit_transform(X_var_train_inter)
X_var_test_scaled = scaler.transform(X_var_test_inter)
#-----------------------------------------------------
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

feature_names = poly.get_feature_names_out(all_x_cols)
coefficients = ridge_model.coef_

coef_df = pd.DataFrame({'coef': coefficients, 'feature': feature_names})
print(coef_df)
plt.show()


