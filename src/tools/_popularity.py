import numpy as np
from ensure import ensure_annotations


@ensure_annotations
def group_items_based_popularity(user_item: np.ndarray, proportions: list):
    assert np.isclose(sum(proportions), 1.0)
    n_items = user_item.shape[1]

    # for the cases that user item is not 0 and 1
    user_item = user_item > 0

    item_popularity = user_item.sum(axis=0, keepdims=False) 
    total_popularity = item_popularity.sum()

    arg_sort = item_popularity.argsort()[::-1]
    sorted_popularity = item_popularity[arg_sort]

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


def calculate_pc_qc(user_item: np.ndarray, rec_matrix: np.ndarray, item_groups: np.ndarray):
    if user_item.shape[1] != rec_matrix.shape[1]:
        raise ValueError("The number of items should be the same in user_item and rec_matrix")

    if user_item.shape[1] != item_groups.shape[0]:
        raise ValueError("the length of item_groups should be same the number of items ")

    unique_groups = np.unique(item_groups)
    n_groups = len(unique_groups)
    num_orig_users, _ = user_item.shape
    num_rec_users, _ = rec_matrix.shape

    p_c_given_u = np.zeros((num_orig_users, n_groups))
    q_c_given_u = np.zeros((num_rec_users, n_groups))

    user_profile_weight = user_item.sum(axis=1)
    user_rec_weight = rec_matrix.sum(axis=1)
    for group_idx, group_id in enumerate(unique_groups):
        group_mask = item_groups == group_id
        true_popularity_per_user = user_item[:, group_mask]
        pred_popularity_per_user = rec_matrix[:, group_mask]

        p_c_numerator = true_popularity_per_user.sum(axis=1)
        p_c_group = p_c_numerator / user_profile_weight

        q_c_numerator = pred_popularity_per_user.sum(axis=1)
        q_c_group = q_c_numerator / user_rec_weight

        p_c_given_u[:, group_idx] = p_c_group
        q_c_given_u[:, group_idx] = q_c_group

    return p_c_given_u, q_c_given_u
