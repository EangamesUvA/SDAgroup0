import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_FILENAME = "data/ESS11e04_0-subset.csv"
df = pd.read_csv(DATA_FILENAME, quotechar='"')

missing_codes = {
    'nwspol': [7777, 8888, 9999],
    'netustm': [6666, 7777, 8888, 9999],
    'ppltrst': [77, 88, 99],
    'pplfair': [77, 88, 99],
    'pplhlp': [77, 88, 99],
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
    "ppltrst": 'most people cant be trusted',
    "pplfair": 'most people try to take advantage of you, or try to be fair',
    "pplhlp": 'people try to be helpful or look out for themselves',
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

    #plt.show()


    plt.figure()
    sns.boxplot(x='ppltrst',y= 'nwspol', data=df_clean)
    plt.xlabel('people cant be trusted')
    plt.ylabel('news/current affairs minutes/day')

    plt.figure()
    sns.boxplot(x='pplfair', y = 'nwspol', data= df_clean)
    plt.xlabel('people try to take advantage of you')
    plt.ylabel('news/current affairs minutes/day')

    plt.figure()
    sns.boxplot(x = 'pplhlp', y = 'nwspol', data = df_clean)
    plt.xlabel('people try to be helpfull or look out for themselves')
    plt.ylabel('news/current affairs minutes/day')

    plt.figure()
    sns.boxplot(x = 'ppltrst', y = 'netustm', data=df_clean)
    plt.xlabel('people cant be trusted')
    plt.ylabel('internet usage minutes/day')

    plt.figure()
    sns.boxplot(x='pplfair', y = 'netustm', data= df_clean)
    plt.xlabel('people try to take advantage of you')
    plt.ylabel('internet usage minutes/day')

    plt.figure()
    sns.boxplot(x = 'pplhlp', y = 'netustm', data = df_clean)
    plt.xlabel('people try to be helpfull or look out for themselves')
    plt.ylabel('internet usage minutes/day')

    plt.show()



if __name__ == "__main__":
    show_plots()
