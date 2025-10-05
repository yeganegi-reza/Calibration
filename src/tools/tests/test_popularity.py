import pytest
import numpy as np
from scipy.sparse import csr_matrix

from src.tools import group_items_based_popularity
from src.tools import calculate_pc
from src.tools import calc_item_popularity

interaction_data = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    ]
)

rec_matrix = np.array(
    [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    ]
)


def test_calc_item_popularity():
    expected_result = np.array([9, 8, 2, 5, 3, 3, 1, 4, 2, 1])
    popularity = calc_item_popularity(interaction_data)
    pytest.approx(expected_result) == popularity


def test_calc_item_popularity_sparce():
    expected_result = np.array([9, 8, 2, 5, 3, 3, 1, 4, 2, 1])
    sparse_csr = csr_matrix(interaction_data)
    popularity = calc_item_popularity(sparse_csr)
    assert popularity.shape == expected_result.shape
    pytest.approx(expected_result) == popularity


def test_group_items_validate():
    results = np.array([0, 0, 1, 1, 1, 1, 2, 1, 1, 2])
    popularity = np.array([9, 8, 2, 5, 3, 3, 1, 4, 2, 1])
    proportions = [0.45, 0.50, 0.05]
    groups = group_items_based_popularity(popularity, proportions)
    assert pytest.approx(results) == groups


def test_calculate_pc():
    groups = np.array([0, 0, 2, 1, 1, 1, 2, 1, 1, 2])
    q_c_target = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.4, 0.2]])
    p_c_target = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.4, 0.6, 0.0],
            [0.66666667, 0.33333333, 0.0],
            [0.25, 0.5, 0.25],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.66666667, 0.33333333, 0.0],
            [0.0, 0.0, 1.0],
            [0.4, 0.4, 0.2],
        ]
    )
    pc = calculate_pc(interaction_data, groups)
    qc = calculate_pc(rec_matrix, groups)

    assert pytest.approx(qc) == q_c_target
    assert pytest.approx(pc) == p_c_target


def test_calculate_zero_intraction():
    groups = np.array([0, 0, 2, 1, 1, 1, 2, 1, 1, 2])
    q_c_target = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.0, 0.0, 0.0]])
    new_rec_matrix = np.array(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    qc = calculate_pc(new_rec_matrix, groups)
    assert pytest.approx(qc) == q_c_target
