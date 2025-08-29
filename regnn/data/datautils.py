import numpy as np
import torch


def train_test_split(num_elems, train_ratio=0.7, seed=42):
    # create and shuffle indices
    indices = np.arange(num_elems)
    # Set the seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)

    # compute number of data for train and test
    train_size = int(train_ratio * num_elems)

    # split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return train_indices, test_indices


def train_test_val_split(
    num_elems, train_ratio=0.5, test_ratio=0.25, val_ratio=0.25, seed=42
):
    # create and shuffle indices
    indices = np.arange(num_elems)
    # Set the seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)

    # compute number of data for train and test
    train_size = int(train_ratio * num_elems)
    test_size = int(test_ratio * num_elems)

    # split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size : train_size + test_size]
    val_indices = indices[train_size + test_size :]

    return train_indices, test_indices, val_indices


def kfold_split(num_elems, k=5, holdout_fold=0, seed=42):
    """
    Split data indices into train and test sets for k-fold cross validation.

    Args:
        num_elems (int): Total number of samples.
        k (int): Number of folds.
        holdout_fold (int): Which fold to use as the test set (0 <= holdout_fold < k).
        seed (int): Random seed for reproducibility.

    Returns:
        train_indices (np.ndarray): Indices for training set.
        test_indices (np.ndarray): Indices for test/validation set.
    """
    if k <= 1:
        raise ValueError("k must be at least 2 for k-fold cross validation.")
    if not (0 <= holdout_fold < k):
        raise ValueError(f"holdout_fold must be between 0 and {k-1}.")

    # create and shuffle indices
    indices = np.arange(num_elems)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # split into k folds
    folds = np.array_split(indices, k)

    # select test and train folds
    test_indices = folds[holdout_fold]
    train_indices = np.concatenate([folds[i] for i in range(k) if i != holdout_fold])

    return train_indices, test_indices


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats
