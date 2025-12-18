import itertools
import pandas as pd
import statsmodels.api as sm
from cleaning_file import *

DATA_FILENAME = "data/all_var_set.csv"

# ========================================
# 0. Load and clean data
# ========================================
cleaner = ESSDataCleaner(DATA_FILENAME)
df_clean = cleaner.full_clean()


def basic_feature_selector(df, predictors, outcomes, p_threshold=0.05):
    """
    Basic feature selector:
    - Creates all pairwise interactions among predictors efficiently
    - Fits a full OLS for each outcome
    - Keeps features with p < threshold
    - Returns selected features and fitted models

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    predictors : list
        List of predictor column names
    outcomes : list
        List of dependent variable names
    p_threshold : float
        Significance level to keep a feature

    Returns
    -------
    results : dict
        Dictionary with keys as outcome names, values as tuples:
        (selected_features, fitted_model)
    """
    results = {}

    # Base predictors DataFrame
    X_base = df[predictors].copy()

    # Efficiently create all pairwise interactions at once
    interactions_dict = {
        f"{a}_x_{b}": df[a] * df[b]
        for a, b in itertools.combinations(predictors, 2)
    }
    interactions = pd.DataFrame(interactions_dict, index=df.index)

    # Combine base and interaction features
    X_full = pd.concat([X_base, interactions], axis=1)

    # Add constant for intercept
    X_full = sm.add_constant(X_full)

    for outcome in outcomes:
        y = df[outcome]

        # Fit full model
        full_model = sm.OLS(y, X_full).fit()

        # Keep features with p < threshold
        significant_features = full_model.pvalues[
            full_model.pvalues < p_threshold
        ].index.tolist()
        if 'const' not in significant_features:
            significant_features.insert(0, 'const')

        # Fit final model with selected features
        final_model = sm.OLS(y, X_full[significant_features]).fit()

        print(
            f"Outcome: {outcome} | Full model R²: {full_model.rsquared:.4f} | "
            f"Basic feature-selector model R²: {final_model.rsquared:.4f}"
        )

        results[outcome] = (significant_features, final_model)

    return results


predictors = [
    'agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl',
    'nwspol', 'netustm', 'vote', 'gndr', 'polintr',
    'feethngr', 'pplfair', 'ppltrst', 'pplhlp',
    'stflife', 'stfeco', 'stfgov'
]

outcomes = ['trstplt']

results = basic_feature_selector(df_clean, predictors, outcomes)

# Access selected features and models for trstplt
selected_features_trstplt, model_trstplt = results['trstplt']
print("Selected features for trstplt:", selected_features_trstplt)
print(model_trstplt.summary())
