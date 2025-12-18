from dataset import *
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_plots():
    for col in numeric_cols:
        plt.figure()
        DATASET.get_columns(col).value_counts().sort_index().plot(kind='bar')
        plt.title(mapping.get(col, col))
        plt.xlabel(mapping.get(col, col))
        plt.ylabel('Count')
    plt.show()


# ------------------------------
# Scatter Plots
# ------------------------------

def scatter_plot(var1_dict, var2_dict):
    for var1, label1 in var1_dict.items():
        for var2, label2 in var2_dict.items():
            plt.figure()
            plt.scatter(DATASET.get_columns(var1), DATASET.get_columns(var2))
            plt.xlabel(label1)
            plt.ylabel(label2)
            plt.title(f"{label2} vs {label1}")
            plt.grid(True)

    plt.show()


def box_plots(var1_dict, var2_dict):
    for indep in INDEP_VAR:
        indep_label = MAPPING[indep]
        for dep in DEP_VAR:
            dep_label = MAPPING[dep]
            plt.figure(figsize=(10, 5))
            DATASET.data.boxplot(column=dep, by=indep)
            plt.title(f"{dep_label} by {indep_label}")
            plt.suptitle("")  # remove default subtitle
            plt.xlabel(indep_label)
            plt.ylabel(dep_label)
            plt.xticks(rotation=45)
            plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_plots()
    scatter_plots(INDEP_VAR, DEP_VAR)
    box_plots(INDEP_VAR, DEP_VAR)

    import seaborn as sns

    for indep in INDEP_VAR:
        indep_label = MAPPING[indep]
        for dep in DEP_VAR:
            dep_label = MAPPING[dep]
            plt.figure(figsize=(10, 5))
            sns.violinplot(x=DATASET.get_columns(indep),
                           y=DATASET.get_columns(dep))
            plt.title(f"{dep_label} by {indep_label}")
            plt.xlabel(indep_label)
            plt.ylabel(dep_label)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
