from data_visualize import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def use_random_forest():
    X = DATASET.get_columns(INDEP_VARS)
    y = DATASET.get_columns(DEP_VARS)

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
