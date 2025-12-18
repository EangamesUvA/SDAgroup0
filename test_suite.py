from dataset import *
import numpy as np
import pytest


# The dataset loaded in as fixture
@pytest.fixture
def dataset():
    return DATASET.data


# Testing if gender is close 50%
def test_gender_distribution(dataset):
    ds = list(dataset["gndr"])
    male = ds.count(1)
    female = ds.count(2)
    no_answer = ds.count(9)
    assert no_answer == 0
    assert abs(0.5 - male / (male + female)) < 0.05
    assert abs(0.5 - female / (male + female)) < 0.05
