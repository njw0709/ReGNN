import numpy as np
import torch

np.random.seed(0)


def train_test_split(num_elems, train_ratio=0.7):
    # create and shuffle indices
    indices = np.arange(num_elems)
    np.random.shuffle(indices)

    # compute number of data for train and test
    train_size = int(train_ratio * num_elems)

    # split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return train_indices, test_indices


def train_test_val_split(num_elems, train_ratio=0.5, test_ratio=0.25, val_ratio=0.25):
    # create and shuffle indices
    indices = np.arange(num_elems)
    np.random.shuffle(indices)

    # compute number of data for train and test
    train_size = int(train_ratio * num_elems)
    test_size = int(test_ratio * num_elems)

    # split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size : train_size + test_size]
    val_indices = indices[train_size + test_size :]

    return train_indices, test_indices, val_indices


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
