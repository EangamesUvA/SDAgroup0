import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd          # for data handling
import numpy as np           # for numeric operations
import networkx as nx        # for network creation
import matplotlib.pyplot as plt  # for plotting the network
from scipy.stats import spearmanr  # for Spearman correlation manually



DATA_FILENAME = "data/all_var_set.csv"
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
    'edlvmenl': [5555, 7777, 8888, 9999],
    'feethngr': [7,8,9,],
    'crmvct': [7,8,9],
    'aesfdrk': [7,8,9],
    'polintr': [7,8,9],
    'trplcnt': [7,8,9],
    'trplcmw': [4,6,7,8,9],
    'stflife': [77,88,99],
    'stfeco': [77,88,99],
    'stfgov': [77,88,99],
    'pplfair': [77,88,99],
    'pplhlp': [77,88,99],
    'ppltrst': [77,88,99]
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
    'feethngr': 'Feel part of same race or ethnic group as most people in country',
    'crmvct': 'Respondent or household member victim of burglary/assault last 5 years',
    'aesfdrk': 'Feeling of safety of walking alone in local area after dark',
    'polintr': 'How interested in politics ',
    'trplcnt': 'How fair the police in [country] treat women/men',
    'trplcmw': 'Unfairly treated by the police because being a man/woman',
    'stflife': 'How satisfied with life as a whole',
    'stfeco': 'How satisfied with present state of economy in country',
    'stfgov': 'How satisfied with the national government',
    'pplfair': "Most people are fair",
    'ppltrst': "Most people can be trusted",
    'pplhlp': 'Most people are trying to help '
}

# Replace missing codes
df.replace(missing_codes, np.nan, inplace=True)

# Drop rows with NaNs
df_clean = df.dropna(subset=list(missing_codes.keys())).copy()

numeric_cols = list(mapping.keys())
df_clean[numeric_cols] = df_clean[numeric_cols].astype(int)


dependent_vars = ['trstplc', 'trstplt']
predictors = ['agea', 'hinctnta', 'edlvenl', 'edlvfenl', 'edlvmenl', 
              'nwspol', 'netustm', 'vote', 'gndr', 'polintr', 
              'crmvct', 'feethngr', 'pplfair', 'ppltrst', 'pplhlp']

# Function to calculate VIF
def calculate_vif(df, features):
    X = sm.add_constant(df[features])
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Run regression for each dependent variable
for dep in dependent_vars:
    X = sm.add_constant(df_clean[predictors])
    y = df_clean[dep]
    model = sm.OLS(y, X).fit(cov_type='HC3')  # robust standard errors
    print(f"\n=== Regression for {dep} ===")
    print(model.summary())
    print("\nVIF:")
    print(calculate_vif(df_clean, predictors))


X = sm.add_constant(X)
model = sm.OLS(y, X).fit(cov_type='HC3')
print(model.summary())

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Select only dependent + independent variables
variables = ['trstplc', 'trstplt', 'agea', 'nwspol', 'netustm', 
             'vote', 'gndr', 'polintr', 'crmvct', 'feethngr', 
             'pplfair', 'ppltrst', 'pplhlp']

df_net = df_clean[variables]

# Spearman correlation matrix
corr_matrix = df_net.corr(method='spearman')

# Build network
G = nx.Graph()
for var in corr_matrix.columns:
    G.add_node(var)

# Add edges above threshold (e.g., |rho| > 0.2)
threshold = 0.2
for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i != j and abs(corr_matrix.loc[i, j]) > threshold:
            G.add_edge(i, j, weight=corr_matrix.loc[i, j])

# Draw network
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
weights = [abs(e[2]['weight'])*3 for e in edges]
edge_colors = ['green' if e[2]['weight'] > 0 else 'red' for e in edges]

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, width=weights, edge_color=edge_colors)
plt.show()

