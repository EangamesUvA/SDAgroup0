# cleaning_file.py
import pandas as pd
import numpy as np

# ========================================
# Missing Codes Dictionary
# ========================================
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
    'feethngr': [7,8,9],
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

# ========================================
# Mapping Dictionary
# ========================================
mapping = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netustm": 'Internet use/day in minutes',
    "trstplc": 'Trust in the police',
    "trstplt": 'Trust in politicians',
    "vote": 'Voted in the last election',
    "gndr": 'Gender/Sex',
    "agea": 'Age of respondent',
    "edlvenl": 'Highest level education Netherlands',
    "hinctnta": 'Household net income',
    "edlvfenl": 'Father education level',
    "edlvmenl": 'Mother education level',
}

# ========================================
# ESS Data Cleaner Class
# ========================================
class ESSDataCleaner: 
    def __init__(self, filename, missing_codes_dict=None, mapping_dict=None):
        """Initialise the data cleaner."""
        self.filename = filename
        self.missing_codes = missing_codes_dict if missing_codes_dict is not None else missing_codes
        self.mapping = mapping_dict if mapping_dict is not None else mapping
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
        if not isinstance(self.missing_codes, dict):
            raise TypeError("missing_codes must be a dictionary of column:list_of_missing_codes")
        self.df.replace(self.missing_codes, np.nan, inplace=True)
        return self.df

    def drop_missing(self, subset=None):
        """Drop rows that contain NaN in certain columns."""
        subset = subset if subset else list(self.missing_codes.keys())
        self.df_clean = self.df.dropna(subset=subset).copy()
        return self.df_clean

    def convert_numeric(self, columns=None):
        """Convert specified columns to integers for analysis."""
        columns = columns if columns else list(self.mapping.keys())
        self.df_clean[columns] = self.df_clean[columns].astype(int)
        return self.df_clean

    def rename_columns(self):
        """Rename columns using the mapping dictionary."""
        if self.df_clean is None:
            raise ValueError("Data not cleaned yet. Run full_clean() first.")
        self.df_clean.rename(columns=self.mapping, inplace=True)
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
