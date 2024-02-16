import numpy as np

# np.random.seed(0)


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
