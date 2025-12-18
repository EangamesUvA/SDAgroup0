from dataset import *
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


print(DATASET.data.shape)

unique_pairs = list(combinations(INDEP_VAR, 2))

interaction_cols = [
    f"{a}_{b}_interaction"
    for a, b in combinations(INDEP_VAR, 2)
]

DATASET.set_columns_interaction(interaction_cols)

all_x_cols = INDEP_VAR + interaction_cols
DATASET.set_columns_to_float(all_x_cols)


vif = VIF_check(DATASET.data, predictors=["stfgov",
                                          "ppltrst_pplhlp_interaction",
                                          "agea_polintr_interaction",
                                          "agea_stfeco_interaction",
                                          "vote_agea_interaction",
                                          "stfeco_stfgov_interaction",
                                          "vote_edlvenl_interaction"
                                          ])

print(vif)
