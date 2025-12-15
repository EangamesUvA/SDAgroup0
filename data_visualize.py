import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -===========- #
#   Filenames   #
# -===========- #
FN_DATASET = "data/ESS11e04_0-subset.csv"


class Data:
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

    columns = list(mapping.keys())

    def __init__(self, filename):
        self.filename = filename
        self.raw_data = pd.read_csv(filename, quotechar='"')
        self.raw_data.replace(Data.missing_codes)

        # Clean the data
        self.data = self.raw_data.dropna(
            subset=list(Data.missing_codes.keys())).copy()
        self.data[Data.columns] = self.data[Data.columns].astype(int)

    def get_columns(self, columns):
        return self.data.loc[:, columns]

    # -=====================================================- #
    #   Variables:                                            #
    #    - dependent: the dependent variables (columns)       #
    #    - independent: the independent variables (columns)   #
    #    - alpha=0.25: the percentage of training data        #
    #   Return values:                                        #
    #    - data_dep_train, data_indep_train,                  #
    #      data_dep_test, data_indep_test                     #
    # -=====================================================- #
    def get_training_set(self, dependent: list[str], independent: list[str], alpha: float=0.75):
        data_dep = self.get_columns(dependent).to_numpy()
        data_indep = self.get_columns(independent).to_numpy()
        len_dep = int(len(data_dep) * alpha)
        len_indep = int(len(data_indep) * alpha)

        return data_dep[:len_dep], data_indep[:len_indep], \
            data_dep[len_dep:], data_indep[len_indep:]


DATASET = Data(FN_DATASET)

DEP_VARS = [
    "trstplc",
    "trstplt",
]

INDEP_VARS = [
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
