import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_FILENAME = "data/ESS11e04_0-subset (1).csv"
df = pd.read_csv(DATA_FILENAME, quotechar='"')

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

# Replace missing codes
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaNs
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

numeric_cols = list(mapping.keys())
df_clean[numeric_cols] = df_clean[numeric_cols].astype(int)


# Plot bar charts
def show_plots():
    for col in numeric_cols:
        plt.figure()
        df_clean[col].value_counts().sort_index().plot(kind='bar')
        plt.title(mapping.get(col, col))
        plt.xlabel(mapping.get(col, col))
        plt.ylabel('Count')
    plt.show()


#------------------------------
#Scatter Plots 
#------------------------------

independent_var = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netustm": 'internet use/day in minutes',
    "vote": 'Voted in the last election',
    "gndr": 'Gender/Sex',
    "agea": 'Age of respondent, calculated',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Households total net income, all sources',
    "edlvfenl": 'Fathers highest level of education, Netherlands',
    "edlvmenl": 'Mothers highest level of education, Netherlands',
}

dependent_var = {
    "trstplc": 'trust in the police',
    "trstplt": 'trust in politicians',
}

def scatter_plot(var1_dict, var2_dict):
    for var1, label1 in var1_dict.items():
        for var2, label2 in var2_dict.items():
            plt.figure()
            plt.scatter(df_clean[var1], df_clean[var2])
            plt.xlabel(label1)
            plt.ylabel(label2)
            plt.title(f"{label2} vs {label1}")
            plt.grid(True)

    plt.show()

def box_plots(var1_dict, var2_dict):
    for indep, indep_label in independent_var.items():
        for dep, dep_label in dependent_var.items():
            plt.figure(figsize=(10, 5))
            df_clean.boxplot(column=dep, by=indep)
            plt.title(f"{dep_label} by {indep_label}")
            plt.suptitle("")  # remove default subtitle
            plt.xlabel(indep_label)
            plt.ylabel(dep_label)
            plt.xticks(rotation=45)
            plt.tight_layout()
    plt.show()

box_plots(independent_var, dependent_var)

import seaborn as sns

for indep, indep_label in independent_var.items():
    for dep, dep_label in dependent_var.items():
        plt.figure(figsize=(10,5))
        sns.violinplot(x=df_clean[indep], y=df_clean[dep])
        plt.title(f"{dep_label} by {indep_label}")
        plt.xlabel(indep_label)
        plt.ylabel(dep_label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
