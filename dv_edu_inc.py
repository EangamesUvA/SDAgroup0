from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
import data_visualize_pandas as dvp
import matplotlib.pyplot as plt
import numpy as np

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
            ddep = dvp.df_clean[dep].to_numpy()
            dindep = dvp.df_clean.loc[:, indep].to_numpy()
            score = use_L2_regressor(
                list(ddep[:train_count]), list(dindep[:train_count]),
                list(ddep[train_count:]), list(dindep[train_count:])
            )
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


if __name__ == "__main__":
    show_plots()
