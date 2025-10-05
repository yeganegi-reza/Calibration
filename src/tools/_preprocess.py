import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from reytools.logger import logging


def generate_user_item_matrix(
    dataset: pd.DataFrame, user_id_map: dict, item_id_map: dict
) -> tuple[csr_matrix, dict, dict]:
    """
    Transforms a DataFrame of user-item interactions (long format) into
    a memory-efficient, sparse User x Item rating matrix (CSR format).

    Args:
        dataset (pd.DataFrame): DataFrame containing 'user', 'item', and 'rating'.

    Returns:
        tuple: (The sparse matrix, user_id_map, item_id_map).
    """
    has_duplicates = dataset.duplicated(subset=["user", "item"]).any()
    if has_duplicates:
        raise ValueError("The combination of user, item should be unique")

    # Apply the mappings to create new indexed columns for matrix construction.
    number_user = len(user_id_map)
    number_item = len(item_id_map)

    user_indices = dataset["user"].map(user_id_map).to_numpy()
    item_indices = dataset["item"].map(item_id_map).to_numpy()
    ratings = dataset["rating"].to_numpy()

    user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(number_user, number_item))

    return user_item_matrix


def create_user_item_map(dataset):
    number_user = dataset["user"].nunique()
    number_item = dataset["item"].nunique()

    user_id_map = dict(zip(dataset["user"].unique(), np.arange(number_user)))
    item_id_map = dict(zip(dataset["item"].unique(), np.arange(number_item)))

    return user_id_map, item_id_map


def k_core_filtering(inter_data: pd.DataFrame, k_item: int, k_user: int) -> pd.DataFrame:
    """
    Performs k-core filtering on the interaction dataset to enforce minimum engagement constraints.

    This function iteratively removes items with fewer than 'k_item' user interactions
    and users with fewer than 'k_user' item interactions. The process continues until
    a stable core is reached where all remaining items and users satisfy their respective
    minimum occurrence thresholds.

    Args:
        inter_data (pd.DataFrame): Interaction data with 'user' and 'item' columns.
        k_item (int): Minimum number of users required per item.
        k_user (int): Minimum number of items required per user.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    while True:
        start_size = len(inter_data)

        # Item pass
        item_counts = inter_data.item.value_counts()
        item_above = set(item_counts[item_counts >= k_item].index)
        inter_data = inter_data[inter_data.item.isin(item_above)]
        print("Records after item pass: ", len(inter_data))
        logging.info(
            f"""Remaining items: {inter_data.item.nunique()}, Remaining users: {inter_data.item.nunique()} """
        )

        # User pass
        user_counts = dataset.user.value_counts()
        user_above = set(user_counts[user_counts >= k_user].index)
        dataset = dataset[dataset.user.isin(user_above)]
        logging.info(
            f"""Remaining items: {inter_data.item.nunique()}, Remaining users: {inter_data.item.nunique()} """
        )

        if len(dataset) == start_size:
            print("Exiting...")
            break
    return dataset
