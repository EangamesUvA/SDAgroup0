from requirements import *
import pytest

@pytest.fixture
def sample_data():
    return np.array([1, 2, 3])

@pytest.fixture
def sample_output():
    return np.array([4, 8, 12])

def test_test(sample_data, sample_output):
    assert np.all([sample_data * 4, sample_output])
