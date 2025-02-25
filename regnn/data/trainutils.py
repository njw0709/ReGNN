import numpy as np
import torch
from pyro.infer import Predictive

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


def get_mse_loss(
    model,
    interaction_predictors,
    interactor,
    controlled_predictors,
    outcome,
    guide,
    n_samples=100,
):
    model.eval()
    mseLoss = torch.nn.MSELoss()
    predictive = Predictive(
        model, guide=guide, num_samples=n_samples, return_sites=("obs", "_RETURN")
    )
    samples = predictive(
        interaction_predictors,
        interactor,
        controlled_predictors,
    )
    pred_summary = summary(samples)
    predicted_epi = pred_summary["_RETURN"]["mean"].squeeze()
    loss_test = mseLoss(predicted_epi, outcome)
    return loss_test.item(), pred_summary
