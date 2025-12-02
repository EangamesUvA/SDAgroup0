import csv
import numpy as np
import matplotlib.pyplot as plt


# Reading in a csv
def read_csv(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
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

DATA_LABELS_MAP = []
for label in DATA_LABELS:
    mapped_label = mapping.get(label, label)
    DATA_LABELS_MAP.append(mapped_label)


def get_data(label_name):
    """
    returns all data for a specific label,
    use the original label name NOT the mapped one
    """
    index = DATA_LABELS.index(label_name)
    data = [person[index] for person in DATASET]
    return data


def make_integer(data):
    """
    turns list of strings into integers
    """
    integer_data = [int(datapoint) for datapoint in data]
    return integer_data


news_minutes_day = get_data('nwspol')
internet_use_freq = get_data('netusoft')
internet_use_day_min = get_data('netustm')
trust_people_general = get_data('ppltrst')
people_take_advantage = get_data('pplfair')
people_helpful = get_data('pplhlp')
gender = get_data('gndr')
highest_education = get_data('edlvenl')
house_hold_income = get_data('hinctnta')
fathers_education = get_data('edulvlfb')
mothers_eduction = get_data('edulvlmb')

plt.figure()
plt.hist(make_integer(news_minutes_day))
plt.title('news minutes/day')

plt.figure()
plt.hist(make_integer(internet_use_freq))
plt.title('internet usage frequency')

plt.figure()
plt.hist(make_integer(internet_use_day_min))
plt.title('internet usage day min')

plt.figure()
plt.hist(make_integer(trust_people_general))
plt.title('trust people general')

plt.figure()
plt.hist(make_integer(people_take_advantage))
plt.title('people take advantage')

plt.figure()
plt.hist(make_integer(people_helpful))
plt.title('people are helpful')

plt.figure()
plt.hist(make_integer(gender))
plt.title('gender')

plt.figure()
plt.hist(make_integer(highest_education))
plt.title('highest education')

plt.figure()
plt.hist(make_integer(house_hold_income))
plt.title('house hold income')

plt.figure()
plt.hist(make_integer(fathers_education))
plt.title('fathers education')

plt.figure()
plt.hist(make_integer(mothers_eduction))
plt.title('mothers education')

plt.show()
