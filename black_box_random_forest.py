from dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


SELECTED_FEATURES = [
    "stfgov",
    "ppltrst_pplhlp_interaction",
    "agea_polintr_interaction",
    "agea_stfeco_interaction",
    "vote_agea_interaction",
    "stfeco_stfgov_interaction",
    "vote_edlvenl_interaction"
]


class Generator:
    def __init__(self, indep_vars, dep_vars) -> None:
        self.X = DATASET.get_columns(indep_vars)
        self.y = DATASET.get_columns(dep_vars)

        self.numerical_cols = self.X.select_dtypes(
            exclude=['object']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(
            include=['object']).columns.tolist()

        self.feature_names = self.numerical_cols + self.categorical_cols
        label_encoder = LabelEncoder()

        self.x_categorical = self.X.select_dtypes(
            include=['object']).apply(label_encoder.fit_transform)
        self.x_numerical = self.X.select_dtypes(exclude=['object']).values

        self.x = pd.concat([
            pd.DataFrame(self.x_numerical), self.x_categorical
        ], axis=1).values

        X_encoded = self.X.copy()

        for col in self.categorical_cols:
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

        self.x = X_encoded.values
        self.y = self.y.squeeze()
        self.feature_names = X_encoded.columns.tolist()


class Regressor:
    def __init__(self, data: Generator) -> None:
        self.regressor = RandomForestRegressor(
            n_estimators=10, random_state=0, oob_score=True)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(data.x, data.y, test_size=0.2, random_state=0)

        self.regressor.fit(self.x_train, self.y_train)

        self.oob_score = self.regressor.oob_score_
        print(f'Out-of-Bag Score: {self.oob_score}')

        self.predictions = self.regressor.predict(self.x_test)

        self.mse = mean_squared_error(self.y_test, self.predictions)
        print(f'Mean Squared Error: {self.mse}')

        self.r2 = r2_score(self.y_test, self.predictions)
        print(f'R-squared: {self.r2}')

        self.X_grid = np.arange(min(data.X.values[:, 0]),
                                max(data.X.values[:, 0]), 0.01).reshape(-1, 1)

        self.X_grid_full = np.tile(data.x.mean(axis=0),
                                   (self.X_grid.shape[0], 1))
        self.X_grid_full[:, 0] = self.X_grid[:, 0]


def plot_predictions_good(regression):
    y_test = regression.y_test

    plt.figure()

    plt.scatter(y_test,
                regression.predictions, label="predictions")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()])

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")

    plt.legend()
    plt.show()


def plot_residual(regression):
    residuals = regression.y_test - regression.predictions

    plt.figure()

    plt.scatter(regression.predictions, residuals)
    plt.axhline(0)

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.legend()
    plt.show()


def plot_feature_importance(data, regression):
    importances = regression.regressor.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()

    plt.barh(
        [data.feature_names[i] for i in indices],
        importances[indices]
    )
    plt.gca().invert_yaxis()

    plt.xlabel("Importance Score")
    plt.title("Random Forest Feature Importances")

    plt.legend()
    plt.show()


def plot_results(data, regression):
    plt.figure()

    plt.scatter(data.X.values[:, 0], data.y,
                color='blue', alpha=0.1, label="Actual Data")
    plt.plot(regression.X_grid[:, 0],
             regression.regressor.predict(regression.X_grid_full),
             color='green', label="Random Forest Prediction")

    plt.title("Random Forest Regression Results")

    plt.legend()
    plt.show()


def show_plots(data, regression):
    plot_predictions_good(regression)
    plot_residual(regression)
    plot_feature_importance(data, regression)
    plot_results(data, regression)


def main(indep_vars, dep_vars):
    data = Generator(indep_vars, dep_vars)

    regression = Regressor(data)

    show_plots(data, regression)


if __name__ == "__main__":
    main(SELECETED_FEATURES, DEP_VAR)
    main(INDEP_VAR, DEP_VAR)
