import numpy as np
import pandas as pd

# ========================================
# Filenames
# ========================================
FN_DATASET = "data/allvariables.csv"

# ========================================
# Missing Codes Dictionary
# ========================================
MISSING_CODES = {
    "aesfdrk": [7, 8, 9],
    "agea": [999],
    "crmvct": [7, 8, 9],
    "edlvenl": [5555, 6666, 7777, 8888, 9999],
    "edlvfenl": [5555, 7777, 8888, 9999],
    "edlvmenl": [5555, 7777, 8888, 9999],
    "feethngr": [7, 8, 9],
    "gndr": [9],
    "hinctnta": [77, 88, 99],
    "netustm": [6666, 7777, 8888, 9999],
    "nwspol": [7777, 8888, 9999],
    "polintr": [7, 8, 9],
    "pplfair": [77, 88, 99],
    "pplhlp": [77, 88, 99],
    "ppltrst": [77, 88, 99],
    "stfeco": [77, 88, 99],
    "stfgov": [77, 88, 99],
    "stflife": [77, 88, 99],
    "trplcmw": [4, 6, 7, 8, 9],
    "trplcnt": [7, 8, 9],
    "trstplc": [77, 88, 99],
    "trstplt": [77, 88, 99],
    "vote": [9],
}

# ========================================
# Mapping Dictionary
# ========================================
MAPPING = {
    "aesfdrk": "Feeling of safety of walking alone in local area after dark",
    "agea": "Age of respondent, calculated",
    "crmvct": "Respondent or household member victim " +
              "of burglary/assault last 5 years",
    "edlvenl": "Highest level education Netherlands",
    "edlvfenl": "Fathers highest level of education, Netherlands",
    "edlvmenl": "Mothers highest level of education, Netherlands",
    "feethngr": "Feel part of same race or ethnic group " +
                "as most people in country",
    "gndr": "Gender/Sex",
    "hinctnta": "Households total net income, all sources",
    "netustm": "internet use/day in minutes",
    "nwspol": "News politics/current affairs minutes/day",
    "polintr": "How interested in politics ",
    "pplfair": "Most people are fair",
    "pplhlp": "Most people are trying to help",
    "ppltrst": "Most people can be trusted",
    "stfeco": "How satisfied with present state of economy in country",
    "stfgov": "How satisfied with the national government",
    "stflife": "How satisfied with life as a whole",
    "trplcmw": "Unfairly treated by the police because being a man/woman",
    "trplcnt": "How fair the police in [country] treat women/men",
    "trstplc": "trust in the police",
    "trstplt": "trust in politicians",
    "vote": "Voted in the last election",
}


class ESSDataCleaner:
    def __init__(self, filename):
        """Initialise the data cleaner."""
        self.filename = filename
        self.df = None
        self.df_clean = None

    def load_data(self, quotechar='"'):
        """Load CSV into a dataframe."""
        self.df = pd.read_csv(self.filename, quotechar=quotechar)
        return self.df

    def replace_missing(self):
        """Replace missing codes with NaN."""
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        if not isinstance(MISSING_CODES, dict):
            raise TypeError("missing_codes must be a dictionary of " +
                            "column:list_of_missing_codes")
        self.df.replace(MISSING_CODES, np.nan, inplace=True)
        return self.df

    def drop_missing(self, subset=None):
        """Drop rows that contain NaN in certain columns."""
        subset = subset if subset else list(MISSING_CODES.keys())
        self.df_clean = self.df.dropna(subset=subset).copy()
        return self.df_clean

    def convert_numeric(self, columns=None):
        """Convert specified columns to integers for analysis."""
        columns = columns if columns else list(MAPPING.keys())
        self.df_clean[columns] = self.df_clean[columns].astype(int)
        return self.df_clean

    def rename_columns(self):
        """Rename columns using the mapping dictionary."""
        if self.df_clean is None:
            raise ValueError("Data not cleaned yet. Run full_clean() first.")
        self.df_clean.rename(columns=MAPPING, inplace=True)
        return self.df_clean

    def full_clean(self, rename_cols=False):
        """Run the full cleaning pipeline."""
        self.load_data()
        self.replace_missing()
        self.drop_missing()
        self.convert_numeric()
        if rename_cols:
            self.rename_columns()
        return self.df_clean


class Data:
    columns = list(MAPPING.keys())

    def __init__(self, filename):
        self.filename = filename

        cleaner = ESSDataCleaner(filename)
        self.data = cleaner.full_clean()

    def get_columns(self, columns):
        existing = [c for c in columns if c in self.data.columns]
        missing = [c for c in columns if c not in self.data.columns]

        df = self.data[existing].copy()

        for col in missing:
            df[col] = self.get_interaction_from_name(col)

        return df

    def get_mult_2_columns(self, column1, column2):
        return self.data[column1] * self.data[column2]

    def get_interaction_from_name(self, name):
        a, b, _ = name.split("_")
        return self.data[a] * self.data[b]

    def set_column(self, column, value):
        self.data[column] = value

    def set_columns_interaction(names):
        values = get_columns(names)
        for name, value in zip(names, values):
            self.set_column(name, value)

    def set_columns_to_float(self, columns):
        self.data[columns] = self.data[columns].astype(float)

    # -=====================================================- #
    #   Variables:                                            #
    #    - dependent: the dependent variables (columns)       #
    #    - independent: the independent variables (columns)   #
    #    - alpha=0.25: the percentage of training data        #
    #   Return values:                                        #
    #    - data_dep_train, data_indep_train,                  #
    #      data_dep_test, data_indep_test                     #
    # -=====================================================- #
    def get_training_set(self, dependent: list[str],
                         independent: list[str], alpha: float = 0.75):
        data_dep = self.get_columns(dependent).to_numpy()
        data_indep = self.get_columns(independent).to_numpy()
        len_dep = int(len(data_dep) * alpha)
        len_indep = int(len(data_indep) * alpha)

        return data_dep[:len_dep], data_indep[:len_indep], \
            data_dep[len_dep:], data_indep[len_indep:]


DATASET = Data(FN_DATASET)

INDEP_VAR = [
    "aesfdrk",
    "agea",
    "crmvct",
    "edlvenl",
    "edlvfenl",
    "edlvmenl",
    "feethngr",
    "gndr",
    "hinctnta",
    "netustm",
    "nwspol",
    "polintr",
    "pplfair",
    "pplhlp",
    "ppltrst",
    "stfeco",
    "stfgov",
    "stflife",
    "trplcmw",
    "trplcnt",
    "vote",
]

DEP_VAR = [
    "trstplt"
]
