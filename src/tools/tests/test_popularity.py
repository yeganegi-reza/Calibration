import pytest
import numpy as np

from src.tools import group_items_based_popularity
from src.tools import calculate_pc_qc

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


def test_group_items_validate():
    results = np.array([0, 0, 1, 1, 1, 1, 2, 1, 1, 2])
    proportions = [0.45, 0.50, 0.05]
    groups = group_items_based_popularity(interaction_data, proportions)
    assert pytest.approx(results) == groups


def test_calculate_pc_qc():
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
    pc, qc = calculate_pc_qc(interaction_data, rec_matrix, groups)
    assert pytest.approx(qc) == q_c_target
    assert pytest.approx(pc) == p_c_target
