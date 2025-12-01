#from requirements import *

import csv
import numpy as np


# Reading in a csv
def read_csv(filename):
    with open(filename, newline='', encoding = 'utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        lines = [row for row in spamreader]
        labels = lines[0]
        stripped_labels = []
        for line in labels:
            stripped_line = line.strip('"')
            stripped_labels.append(stripped_line)

        data = lines[1:]
        stripped_data = []
        for line in data:
            stripped_data_person = []
            for point in line:
                stripped_point = point.strip('"')
                stripped_data_person.append(stripped_point)
            stripped_data.append(stripped_data_person)

    return stripped_labels, stripped_data

DATA_FILENAME = "data/ESS11e04_0-subset.csv"
DATA_LABELS, DATASET = read_csv(DATA_FILENAME)

mapping = {
    "nwspol": 'News politics/current affairs minutes/day',
    "netusoft": 'internet use how often',
    "netustm" : 'internet use/day in minutes',
    "ppltrst" : 'most people cant be trusted',
    "pplfair" : 'most people try to take advantage of you, or try to be fair',
    "pplhlp" : 'people try to be helpful or look out for themselves',
    "gndr" : 'Gender/Sex',
    "edlvenl" : 'Highest level education Netherlands',
    "hinctnta": 'Households total net income, all sources',
    "edulvlfb" : 'Fathers highest level of education',
    "edulvlmb" : 'Mothers highest level of education',
}


DATA_LABELS_MAP = []
for label in DATA_LABELS:
    mapped_label = mapping.get(label,label)
    DATA_LABELS_MAP.append(mapped_label)
