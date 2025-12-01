from requirements import *

import csv
import numpy as np


# Reading in a csv
def read_csv(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lines = [row for row in spamreader]
        labels = lines[0]
        data = lines[1:]
    return labels, data

DATA_FILENAME = "data/ESS11e04_0-subset.csv"
DATA_LABELS, DATASET = read_csv(DATA_FILENAME)
