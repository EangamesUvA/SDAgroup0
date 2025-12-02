# Packages
import numpy as np
import pytest


# Data that can be used as argument to a test function
@pytest.fixture
def sample_data():
    return np.array([1, 2, 3])


# Data that can be used as argument to a test function
@pytest.fixture
def sample_output():
    return np.array([4, 8, 12])


# Test function automatically used by pytest
def test_test(sample_data, sample_output):
    assert np.all([sample_data * 4, sample_output])
