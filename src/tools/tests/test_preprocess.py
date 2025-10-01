import pytest
import numpy as np
import pandas as pd

from tools import generate_user_item_matrix
from tools import create_user_item_map

test_dataframe = pd.DataFrame(
    {
        "user": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        "item": [101, 103, 102, 103, 104, 101, 105, 102, 104, 105, 106, 101, 103, 105, 106],
        "rating": [4, 1, 5, 3, 1, 5, 3, 4, 1, 4, 2, 2, 3, 2, 3],
        "time": [166, 169, 171, 173, 175, 176, 176, 176, 177, 182, 194, 196, 196, 196, 199],
    }
)

user_map_result = {
    np.int64(1): np.int64(0),
    np.int64(2): np.int64(1),
    np.int64(3): np.int64(2),
    np.int64(4): np.int64(3),
    np.int64(5): np.int64(4),
}
item_map_result = {
    np.int64(101): np.int64(0),
    np.int64(103): np.int64(1),
    np.int64(102): np.int64(2),
    np.int64(104): np.int64(3),
    np.int64(105): np.int64(4),
    np.int64(106): np.int64(5),
}


def test_generate_user_item_matrix():
    sparse_matrix = generate_user_item_matrix(test_dataframe, user_map_result, item_map_result)
    result_matrix = np.array(
        [[4, 1, 0, 0, 0, 0], [0, 3, 5, 1, 0, 0], [5, 0, 0, 0, 3, 0], [0, 0, 4, 1, 4, 2], [2, 3, 0, 0, 2, 3]]
    )

    assert pytest.approx(sparse_matrix.toarray()) == result_matrix


def test_create_user_item_map():

    user_map, item_map = create_user_item_map(test_dataframe)
    assert pytest.approx(user_map_result) == user_map
    assert pytest.approx(item_map) == item_map_result
