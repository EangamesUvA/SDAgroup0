# Packages
import numpy as np
import pytest
import data_visualize_pandas as dvp


# Data that can be used as argument to a test function
@pytest.fixture
def sample_data():
    return np.array([1, 2, 3])


# Data that can be used as argument to a test function
@pytest.fixture
def sample_output():
    return np.array([4, 8, 12])


@pytest.fixture
def dataset():
    return dvp.df_clean


# Test function automatically used by pytest
def test_test(sample_data, sample_output):
    assert np.all([sample_data * 4, sample_output])


# Testing if gender is close 50%
def test_gender_distribution(dataset):
    ds = list(dataset["gndr"])
    male = ds.count(1)
    female = ds.count(2)
    no_answer = ds.count(9)
    assert no_answer == 0
    assert abs(0.5 - male / (male + female)) < 0.05
    assert abs(0.5 - female / (male + female)) < 0.05
