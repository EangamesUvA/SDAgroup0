import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def check_linearity(df, continous_vars, dep_var):
    """
    Checks linearity, visually, for continuous variables.
    """
    for var in continous_vars:
        sns.scatterplot(x=var, y=dep_var, data=df)
        sns.regplot(
            x=var,
            y=dep_var,
            data=df,
            scatter=False,
            color='red'
        )
        plt.title(f"Linearity check: {var} vs {dep_var}")
        plt.show()


def VIF_check(df, predictors):
    """
    Computes Variance Inflation Factors (VIF) for a set of predictors.

    Parameters
    ----------
    df : pandas DataFrame
        Cleaned dataframe
    predictors : list
        List of predictor column names

    Returns
    -------
    vif_df : pandas DataFrame
        DataFrame with variables and their VIF values
    """
    X = df[predictors]
    X = sm.add_constant(X)

    vif_df = pd.DataFrame()
    vif_df["variable"] = X.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_df


def plot_qq(model, dep_var_name=None):
    """
    Creates a Q-Q plot for residuals of an OLS model.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResults
        A fitted OLS model.
    dep_var_name : str, optional
        Name of the dependent variable, used for plot title.
    """
    residuals = model.resid
    sm.qqplot(residuals, line='45')
    title = (
        f"Q-Q Plot - {dep_var_name}"
        if dep_var_name else "Q-Q Plot - Residuals"
    )
    plt.title(title)
    plt.show()


def plot_resid_vs_fitted(model, dep_var_name=None):
    """
    Plots residuals vs fitted values for an OLS model to check homoscedasticity.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResults
        A fitted OLS model.
    dep_var_name : str, optional
        Name of the dependent variable, used for plot title.
    """
    fitted = model.fittedvalues
    residuals = model.resid

    sns.scatterplot(x=fitted, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")

    title = (
        f"Residuals vs Fitted - {dep_var_name}"
        if dep_var_name else "Residuals vs Fitted"
    )
    plt.title(title)
    plt.show()


def check_outliers_influence(model, dep_var_name=None):
    """
    Checks for outliers and influential points in an OLS model.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResults
        A fitted OLS model.
    dep_var_name : str, optional
        Name of the dependent variable, used for plot titles.
    """
    influence = model.get_influence()
    standardized_resid = influence.resid_studentized_internal
    cooks = influence.cooks_distance[0]

    n = len(standardized_resid)

    # Plot standardized residuals
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(n), standardized_resid, alpha=0.6)
    plt.axhline(3, color='red', linestyle='--')
    plt.axhline(-3, color='red', linestyle='--')
    plt.xlabel("Observation")
    plt.ylabel("Standardized Residual")
    plt.title(
        f"Standardized Residuals - {dep_var_name}"
        if dep_var_name else "Standardized Residuals"
    )

    # Plot Cook's distance
    plt.subplot(1, 2, 2)
    plt.scatter(range(n), cooks, alpha=0.6)
    plt.axhline(4 / n, color='red', linestyle='--')
    plt.xlabel("Observation")
    plt.ylabel("Cook's Distance")
    plt.title(
        f"Cook's Distance - {dep_var_name}"
        if dep_var_name else "Cook's Distance"
    )

    plt.tight_layout()
    plt.show()

    # Flag extreme points
    extreme_resid_idx = np.where(np.abs(standardized_resid) > 3)[0]
    high_cooks_idx = np.where(cooks > 4 / n)[0]

    print(
        f"{dep_var_name} - Observations with |standardized residual| > 3:",
        extreme_resid_idx
    )
    print(
        f"{dep_var_name} - Observations with Cook's distance > 4/n:",
        high_cooks_idx
    )
