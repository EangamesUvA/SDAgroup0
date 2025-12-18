from dataset import *
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

unique_pairs = list(combinations(INDEP_VAR, 2))

interaction_cols = [
    f"{a}_{b}_interaction"
    for a, b in combinations(INDEP_VAR, 2)
]

DATASET.set_columns_interaction(interaction_cols)
all_x_cols = INDEP_VAR + interaction_cols
DATASET.set_columns_to_float(all_x_cols)


def log_likelihood(y_true, y_pred):
    n = len(y_true)
    residuals = y_true - y_pred
    sigma2 = np.sum(residuals**2)/n
    logL = (-n/2) * np.log(2*np.pi*sigma2) - \
        (1/(2*sigma2)) * np.sum(residuals**2)
    return logL


def calculate_BIC(k, n, logL):
    return k * np.log(n) - 2 * logL


def plot_k_BIC(data_frame, x, y):
    selected_features = []
    features_at_step = []
    remaining_features = x.copy()
    y_values = DATASET.get_columns(y)
    list_k = []
    list_BIC = []

    while remaining_features:
        dict_BIC = {}
        for feature in remaining_features:
            test_feature = selected_features + [feature]
            x_test = data_frame[test_feature]
            model = LinearRegression()
            model.fit(x_test, y_values)
            y_pred = model.predict(x_test)

            logL = log_likelihood(y_values, y_pred)
            k = len(test_feature)
            n = len(y_values)
            BIC = calculate_BIC(k, n, logL)
            dict_BIC[feature] = BIC

        feature_min_BIC = min(dict_BIC, key=dict_BIC.get)
        min_BIC_score = dict_BIC[feature_min_BIC]
        print(f"{min_BIC_score}     ", end="\r")
        selected_features.append(feature_min_BIC)
        remaining_features.remove(feature_min_BIC)
        list_BIC.append(min_BIC_score)
        list_k.append(len(selected_features))
        features_at_step.append(selected_features.copy())

    min_index = np.argmin(list_BIC)
    min_k = list_k[min_index]
    features_minimum = features_at_step[min_index]

    return list_k, list_BIC, min_k, features_minimum


X = all_x_cols
Y_politicians = 'trstplt'
data_frame = DATASET.data

k_politicians, BIC_politicians, min_k_politicians, min_features_politicians = \
    plot_k_BIC(data_frame, X, Y_politicians)


print(f'the selected features for politicians are{min_features_politicians}')

plt.figure()
plt.plot(k_politicians, BIC_politicians)
plt.axvline(min_k_politicians, color='r',
            label=f'min BIC k = {min_k_politicians}')
plt.xlabel('k')
plt.ylabel('BIC')
plt.title('BIC score for k features outcome trust in politicians')
plt.legend()


all_x_cols = min_features_politicians
df_selected_features_outcome = \
    DATASET.get_columns(min_features_politicians + ['trstplt'])

list_rmse = []
list_rsquared = []
list_coef = []
N = 1000
for i in range(N):
    n = len(df_selected_features_outcome)
    bootstrap_sample = df_selected_features_outcome.sample(n=n, replace=True)

    # splits the data into training and testing data
    X_var_train, X_var_test, Y_var_train, Y_var_test = \
        train_test_split(bootstrap_sample[min_features_politicians],
                         bootstrap_sample['trstplt'],
                         test_size=0.3, random_state=42)

    # scales the data so higher absolute numbers
    # like income don't dominate the cost function
    scaler = StandardScaler()
    X_var_train_scaled = scaler.fit_transform(X_var_train)
    X_var_test_scaled = scaler.transform(X_var_test)

    # -----------------------------------------------------
    # create multiple values for alpha (lambda) on different scales (log)
    alphas = np.logspace(-3, 3, 7)
    dict_alpha = {'alpha': alphas}

    # use a grid search to test different alpha's
    # and see which one predicts the "unseen" data best
    grid = GridSearchCV(Ridge(), dict_alpha,
                        scoring='neg_mean_squared_error',
                        cv=5, n_jobs=-1)

    grid.fit(X_var_train_scaled, Y_var_train)

    best_alpha = grid.best_params_['alpha']

    # training the model with the best alpha
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_var_train_scaled, Y_var_train)

    # predicted Y based on unseen test X
    Y_pred = ridge_model.predict(X_var_test_scaled)

    # evaluating performance of the model
    rmse = np.sqrt(mean_squared_error(Y_var_test, Y_pred))
    list_rmse.append(rmse)
    r2 = r2_score(Y_var_test, Y_pred)
    list_rsquared.append(r2)

    coefficients = ridge_model.coef_
    list_coef.append(coefficients)


feature_names = min_features_politicians

coef_df = pd.DataFrame(list_coef, columns=feature_names)
ci_df = pd.DataFrame({
    '2.5%': coef_df.quantile(0.025),
    '97.5%': coef_df.quantile(0.975)
})

print(f'the 95% CI for r2 = {[np.percentile(list_rsquared, 2.5),
                              np.percentile(list_rsquared, 97.5)]}, N = {N}')

print(f'the 95% CI for rmse = {[np.percentile(list_rmse, 2.5),
                                np.percentile(list_rmse, 97.5)]}, N = {N}')

print(f'the 95% CI for the coefficients are {ci_df}, N = {N}')

plt.show()
