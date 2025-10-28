import numpy as np
from ensure import ensure_annotations

from typing import Union
from scipy.sparse import csr_matrix

Matrix = Union[np.ndarray, csr_matrix]


@ensure_annotations
def calc_item_popularity(user_item: Matrix, binary: bool = True) -> np.ndarray:
    # for the cases that user item is not 0 and 1
    if binary:
        user_item = user_item > 0
    if isinstance(user_item, np.ndarray):
        item_popularity = user_item.sum(axis=0, keepdims=False)
    else:
        item_popularity = user_item.sum(axis=0)
        item_popularity = np.array(item_popularity).flatten()
    return item_popularity


@ensure_annotations
def group_items_based_popularity(item_popularity: np.ndarray, proportions: list):
    assert np.isclose(sum(proportions), 1.0)
    n_items = len(item_popularity)

    arg_sort = item_popularity.argsort()[::-1]
    sorted_popularity = item_popularity[arg_sort]

    total_popularity = item_popularity.sum()
    cumulative_proportions = np.cumsum(proportions)
    popularity_boundaries = cumulative_proportions * total_popularity

    # To ensure we assign a group to the last item of the sorted list
    popularity_boundaries[-1] += 0.1

    current_index = 0
    item_group = np.zeros((n_items), dtype=int) - 1
    cumulative_popularity = np.cumsum(sorted_popularity)
    for group_id, boundry in enumerate(popularity_boundaries):
        next_index = np.searchsorted(cumulative_popularity, boundry)
        true_inds = arg_sort[current_index:next_index]
        item_group[true_inds] = group_id
        current_index = next_index

    return item_group


def calculate_pc(user_item: np.ndarray, item_groups: np.ndarray, epsilon: float = 1e-10):
    if user_item.shape[1] != item_groups.shape[0]:
        raise ValueError("the length of item_groups should be same the number of items ")

    unique_groups = np.unique(item_groups)
    n_groups = len(unique_groups)
    num_orig_users, _ = user_item.shape

    p_c_given_u = np.zeros((num_orig_users, n_groups))
    total_user_profile_actions = user_item.sum(axis=1).astype(float)

    # for handling divition by zero
    zero_mask = total_user_profile_actions < epsilon
    total_user_profile_actions[zero_mask] = epsilon
    for group_idx, group_id in enumerate(unique_groups):
            group_mask = item_groups == group_id
            for user_index in range(num_orig_users): 
                true_popularity_per_user = user_item[user_index, group_mask]
                p_c_numerator = true_popularity_per_user.sum()
                p_c_group = p_c_numerator / total_user_profile_actions[user_index]
                p_c_given_u[user_index, group_idx] = p_c_group

    return p_c_given_u
