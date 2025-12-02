import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_FILENAME = "data/ESS11e04_0-subset.csv"
df = pd.read_csv(DATA_FILENAME, quotechar='"')

missing_codes = {
    'nwspol': [7777, 8888, 9999],
    'netusoft': [7, 8, 9],
    'netustm': [6666, 7777, 8888, 9999],
    'ppltrst': [77, 88, 99],
    'pplfair': [77, 88, 99],
    'pplhlp': [77, 88, 99],
    'gndr': [9],
    'edlvenl': [5555, 6666, 7777, 8888, 9999],
    'hinctnta': [77, 88, 99],
    'edulvlfb': [5555, 6666, 7777, 8888, 9999],
    'edulvlmb': [5555, 6666, 7777, 8888, 9999]
}

iscd_mapping = {
    0: 1,      # Not completed ISCED level 1
    113: 2,    # ISCED 1, completed primary education
    129: 3,    # Vocational ISCED 2C < 2 years
    212: 4,    # General/pre-vocational ISCED 2A/2B
    213: 5,    # General ISCED 2A, access ISCED 3A
    221: 6,    # Vocational ISCED 2C >= 2 years
    222: 7,    # Vocational ISCED 2A/2B, access ISCED 3 vocational
    223: 8,    # Vocational ISCED 2, access ISCED 3 general/all
    229: 9,    # Vocational ISCED 3C < 2 years
    311: 10,   # General ISCED 3 >=2 years, no access ISCED 5
    312: 11,   # General ISCED 3A/3B, access ISCED 5B/lower tier 5A
    313: 12,   # General ISCED 3A, access upper tier ISCED 5A/all 5
    321: 13,   # Vocational ISCED 3C >= 2 years, no access ISCED 5
    322: 14,   # Vocational ISCED 3A, access ISCED 5B/ lower tier 5A
    323: 15,   # Vocational ISCED 3A, access upper tier ISCED 5A/all 5
    412: 16,   # General ISCED 4A/4B, access ISCED 5B/lower tier 5A
    413: 17,   # General ISCED 4A, access upper tier ISCED 5A/all 5
    421: 18,   # ISCED 4 programmes without access ISCED 5
    422: 19,   # Vocational ISCED 4A/4B, access ISCED 5B/lower tier 5A
    423: 20,   # Vocational ISCED 4A, access upper tier ISCED 5A/all 5
    510: 21,   # ISCED 5A short,
               # intermediate/academic/general tertiary below bachelor
    520: 22,   # ISCED 5B short, advanced vocational qualifications
    610: 23,   # ISCED 5A medium, bachelor/equivalent from lower tier tertiary
    620: 24,   # ISCED 5A medium,
               # bachelor/equivalent from upper/single tier tertiary
    710: 25,   # ISCED 5A long, master/equivalent from lower tier tertiary
    720: 26,   # ISCED 5A long,
               # master/equivalent from upper/single tier tertiary
    800: 27    # ISCED 6, doctoral degree
}

mapping = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netusoft": 'internet use how often',
    "netustm": 'internet use/day in minutes',
    "ppltrst": 'most people cant be trusted',
    "pplfair": 'most people try to take advantage of you, or try to be fair',
    "pplhlp": 'people try to be helpful or look out for themselves',
    "gndr": 'Gender/Sex',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Households total net income, all sources',
    "edulvlfb": 'Fathers highest level of education',
    "edulvlmb": 'Mothers highest level of education',
}

# Replace missing codes
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaNs
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

# Map ISCED education to ordinal
df_clean['father_education_ordinal'] = df_clean['edulvlfb'].map(
    lambda x: iscd_mapping.get(int(x), np.nan))
df_clean['mother_education_ordinal'] = df_clean['edulvlmb'].map(
    lambda x: iscd_mapping.get(int(x), np.nan))

# Convert all columns to int
numeric_cols = list(mapping.keys()) + \
        ['father_education_ordinal', 'mother_education_ordinal']
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


if __name__ == "__main__":
    show_plots()
