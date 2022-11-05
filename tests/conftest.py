import pytest

@pytest.fixture
def example1_input_data():
    input_data = {'a': {'aa': 0.3, 'cc': 0.7, },
                  'b': {'aa': 2, 'bb': 0.5, },
                  'c': {'aa': 0.5, 'bb': 0.5, 'cc': 1.5, }}
    return input_data
