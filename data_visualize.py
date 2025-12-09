from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -===========- #
#   Filenames   #
# -===========- #
FN_DATASET = "data/ESS11e04_0-subset.csv"


class Data:
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

    columns = list(mapping.keys())

    def __init__(self, filename):
        self.filename = filename
        self.raw_data = pd.read_csv(filename, quotechar='"')
        self.raw_data.replace(Data.missing_codes)

        # Clean the data
        self.data = self.raw_data.dropna(
            subset=list(Data.missing_codes.keys())).copy()
        self.data[Data.columns] = self.data[Data.columns].astype(int)

    def get_columns(self, columns):
        return self.data.loc[:, columns]

    # -=====================================================- #
    #   Variables:                                            #
    #    - dependent: the dependent variables (columns)       #
    #    - independent: the independent variables (columns)   #
    #    - alpha=0.25: the percentage of training data        #
    #   Return values:                                        #
    #    - data_dep_train, data_indep_train,                  #
    #      data_dep_test, data_indep_test                     #
    # -=====================================================- #
    def get_training_set(self, dependent, independent, alpha=0.75):
        data_dep = self.get_columns(dependent).to_numpy()
        data_indep = self.get_columns(independent).to_numpy()
        len_dep = int(len(data_dep) * alpha)
        len_indep = int(len(data_indep) * alpha)

        return data_dep[:len_dep], data_indep[:len_indep], \
            data_dep[len_dep:], data_indep[len_indep:]


DATASET = Data(FN_DATASET)

dep_vars = [
    "trstplc",
    "trstplt",
]

indep_vars = [
    "vote",
    "nwspol",
    "netustm",
    "gndr",
    "agea",
    "edlvenl",
    "hinctnta",
    "edlvfenl",
    "edlvmenl"
]

COMBS = [[], [], [], []]
for i, indep_var in enumerate(indep_vars):
    for j, dep_var in enumerate(dep_vars):
        COMBS[0].append((dep_var, [indep_var]))
        for indep_var2 in indep_vars[i:]:
            COMBS[j+1].append((dep_var, [indep_var, indep_var2]))


def show_bar(percentage):
    amount_total = 50
    complete = round(percentage * amount_total / 100)
    print(" [" + "=" * complete + "-" * (amount_total - complete) + \
            f"] {percentage}%   ", end="\r")


def MODEL(alpha, tol):
    return Ridge(alpha=alpha, tol=tol)


def use_L2_regressor(ddep_train, dindep_train, ddep, dindep):
    ddep_train = np.array(ddep_train).reshape(-1, 1)
    ddep = np.array(ddep).reshape(-1, 1)
    dindep_train = np.array(dindep_train)
    dindep = np.array(dindep)

    cross_val_scores_ridge = []
    Lambda = []

    for i in range(1, 20):
        Model = MODEL(alpha=i * 0.1, tol=0.0925)
        Model.fit(ddep_train, dindep_train)
        scores = cross_val_score(Model, ddep, dindep, cv=10)
        avg_cross_val_score = np.mean(scores) * 100
        cross_val_scores_ridge.append(avg_cross_val_score)
        Lambda.append(i * 0.25)

    l = max(Lambda)
    ModelChosen = MODEL(alpha=l, tol=0.0925)
    ModelChosen.fit(ddep_train, dindep_train)
    return ModelChosen.score(ddep, dindep)


def show_plots():
    train_count = 300
    for i, combs in enumerate(COMBS):
        models = list(map(lambda x: str(x[0]) + "\n" + "/".join(x[1]), combs))
        scores = dict(zip(map(lambda x: x[1][0], combs), [[] for _ in range(len(combs))]))
        labels = []
        done = 0
        for (dep, indep) in combs:
            score = use_L2_regressor(*DATASET.get_training_set(dep, indep))
            scores[indep[0]].append(score)
            if len(indep) > 1:
                scores[indep[1]].append(score)
            labels.append(f"{dep}\nvs\n{indep}")
            done += 1
            show_bar(round(done/len(combs)*100, 1))
        if i == 0:
            plt.bar(models, np.concatenate(np.array(list(scores.values()))))
            plt.xlabel('L2 regression on different combos of dependent/independent variables')
            plt.ylabel('Score')
            plt.show()
        else:
            plt.figure(figsize=(6, 5))
            score_matrix = list(scores.values())

            im = plt.imshow(score_matrix, cmap="viridis", aspect="auto")
            plt.colorbar(im)

            plt.xticks(np.arange(len(indep_vars)), indep_vars, rotation=45, ha="right")
            plt.yticks(np.arange(len(indep_vars)), indep_vars, rotation=45, ha="right")
            plt.xlabel("Independent variable 1")
            plt.ylabel("Independent variable 2")
            plt.title(f"L2 regression on combination of 2 independent variables with {dep} dependent variable")
            plt.show()


def use_random_forest():
    X = DATASET.get_columns(indep_vars)
    y = DATASET.get_columns(dep_vars)

    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    feature_names = numerical_cols + categorical_cols

    label_encoder = LabelEncoder()
    x_categorical = X.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numerical = X.select_dtypes(exclude=['object']).values
    x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

    X_encoded = X.copy()

    for col in categorical_cols:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

    x = X_encoded.values
    y = y.values
    feature_names = X_encoded.columns.tolist()

    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    regressor.fit(x_train, y_train)

    oob_score = regressor.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    predictions = regressor.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y_test, predictions)
    print(f'R-squared: {r2}')

    X_grid = np.arange(min(X.values[:, 0]), max(X.values[:, 0]), 0.01).reshape(-1, 1)

    X_grid_full = np.tile(x.mean(axis=0), (X_grid.shape[0], 1))
    X_grid_full[:, 0] = X_grid[:, 0]

    plt.figure()
    plt.scatter(y_test, predictions, label="predictions")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()])
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.show()

    residuals = y_test - predictions

    plt.figure()
    plt.scatter(predictions, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in indices],
        importances[indices]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Random Forest Feature Importances")
    plt.show()

    plt.figure()
    plt.scatter(X.values[:, 0], y[:, 0], color='blue', alpha=0.1, label="Actual Data")
    plt.plot(X_grid[:, 0], regressor.predict(X_grid_full), color='green', label="Random Forest Prediction")  
    plt.title("Random Forest Regression Results")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    use_random_forest()
    # show_plots()
