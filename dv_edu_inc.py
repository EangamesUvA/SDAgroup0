from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import data_visualize_pandas as dvp
import matplotlib.pyplot as plt
import numpy as np

COMBS = [
    ("ppltrst", ["edlvenl"]),
    ("pplfair", ["edlvenl"]),
    ("pplhlp", ["edlvenl"]),
    ("ppltrst", ["hinctnta"]),
    ("pplfair", ["hinctnta"]),
    ("pplhlp", ["hinctnta"]),
    ("edlvenl", ["hinctnta"]),
    ("ppltrst", ["edlvenl", "hinctnta"]),
    ("pplfair", ["edlvenl", "hinctnta"]),
    ("pplhlp", ["edlvenl", "hinctnta"]),
]

def MODEL(alpha, tol):
    return Ridge(alpha=alpha, tol=tol)


def use_L2_regressor(ddep_train, dindep_train, ddep, dindep):
    ddep_train = np.array(ddep_train).reshape(-1, 1)
    ddep = np.array(ddep).reshape(-1, 1)
    dindep_train = np.array(dindep_train)
    dindep = np.array(dindep)

    cross_val_scores_ridge = []
    Lambda = []

    for i in range(1, 9):
        Model = MODEL(alpha=i * 0.25, tol=0.0925)
        Model.fit(ddep_train, dindep_train)
        scores = cross_val_score(Model, ddep, dindep, cv=10)
        avg_cross_val_score = np.mean(scores) * 100
        cross_val_scores_ridge.append(avg_cross_val_score)
        Lambda.append(i * 0.25)

    for i in range(0, len(Lambda)):
        print(str(Lambda[i]) + ' : ' + str(cross_val_scores_ridge[i]))

    l = max(Lambda)
    ModelChosen = MODEL(alpha=l, tol=0.0925)
    ModelChosen.fit(ddep_train, dindep_train)
    return ModelChosen.score(ddep, dindep)


def show_plots():
    train_count = 300
    models = list(map(lambda x: str(x[0]) + "\n" + "/".join(x[1]), COMBS))
    scores = []
    for (dep, indep) in COMBS:
        ddep = dvp.df_clean[dep].to_numpy()
        dindep = dvp.df_clean.loc[:, indep].to_numpy()
        scores.append(use_L2_regressor(
            list(ddep[:train_count]), list(dindep[:train_count]),
            list(ddep[train_count:]), list(dindep[train_count:])
        ))
    plt.bar(models, scores)
    plt.xlabel('L2 regression on different combos of dependent/independent variables')
    plt.ylabel('Score')
    plt.show()


if __name__ == "__main__":
    show_plots()
