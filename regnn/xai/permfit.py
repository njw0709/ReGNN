import numpy as np
import torch
from math import erf, sqrt
from typing import Optional, List, Dict
from tqdm import tqdm
from regnn.model import ReGNN, IndexPredictionModel


def permfit_feature_importance_index(
    model: IndexPredictionModel,
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 100,
    metric: str = "mse",
    random_state: Optional[int] = None,
    feature_name_list: Optional[List[str]] = None,
    device: str = "cuda",
):
    """
    Compute PermFIT feature importance scores and significance for a regression model.

    Parameters:
        model (torch.nn.Module): Trained PyTorch model for regression.
        X (numpy.ndarray or torch.Tensor): Feature data of shape (n_samples, n_features).
        y (numpy.ndarray or torch.Tensor): True target values of shape (n_samples,) or (n_samples, 1).
        n_permutations (int): Number of random permutations to perform for each feature (default 100).
        metric (str): Performance metric - 'mse' for mean squared error or 'mae' for mean absolute error.
        random_state (int, optional): Random seed for reproducibility of permutations.

    Returns:
        importance_table (list of dict): Each dict corresponds to one feature and contains:
            - 'feature_index': index of the feature (0-based).
            - 'importance': estimated importance score (mean increase in error when permuted).
            - '95% CI': tuple (lower, upper) 95% confidence interval for the importance.
            - 'p_value': p-value testing if importance > 0 (one-sided).
    """
    assert (
        n_permutations > 30
    ), "n_permutation must be higher than 30 to appropriately estimate p-values"
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    model.eval()  # ensure model is in evaluation mode (no dropout, etc.)
    model.to(device)
    if X.ndim == 2:
        X = np.expand_dims(X, axis=1)
    X_tensor = torch.tensor(X, device=device, dtype=torch.float32)
    y_tensor = torch.tensor(y, device=device, dtype=torch.float32)
    y_tensor = y_tensor.view(-1, 1, 1)
    n_features = X_tensor.shape[-1]
    importance_table = []
    # Use NumPy copy of X for efficient column shuffling

    for j in tqdm(range(n_features)):
        # Compute error increase for feature j over multiple random permutations
        perm_errors = []
        for _ in range(n_permutations):
            X_perm = X.copy().astype(np.float32)
            X_perm[:, :, j] = np.random.permutation(
                X_perm[:, :, j]
            )  # shuffle feature j across samples
            X_perm_tensor = torch.from_numpy(X_perm).to(device)
            with torch.no_grad():
                perm_pred = model(X_perm_tensor)
            if metric == "mse":
                err = torch.mean((perm_pred - y_tensor) ** 2).item()
            else:  # 'mae'
                err = torch.mean(torch.abs(perm_pred - y_tensor)).item()
            perm_errors.append(err)
        perm_errors = np.array(perm_errors)
        # Calculate importance: mean increase in error due to permuting feature j
        error_diffs = perm_errors
        importance_score = error_diffs.mean()
        # Confidence interval for the importance (approximate 95% CI for the mean)
        # Using normal approximation: mean ± 1.96 * (std_error)
        std_error = error_diffs.std(ddof=1) / np.sqrt(n_permutations)
        ci_lower = importance_score - 1.96 * std_error
        ci_upper = importance_score + 1.96 * std_error
        # One-sample t-test (one-sided) for H0: importance <= 0 vs H1: importance > 0
        if error_diffs.std(ddof=1) < 1e-12:
            # If all differences are (almost) identical, handle edge cases
            if importance_score <= 0:
                p_val = 1.0  # no evidence of >0 importance
            else:
                p_val = 1.0 / (n_permutations + 1)  # minimal p-value given permutations
        else:
            # Compute t-statistic
            t_stat = importance_score / std_error
            # Degrees of freedom = n_permutations - 1 (if n_permutations is large, t ~ normal)
            # Compute one-sided p-value from t-statistic. For large permutation counts, use normal approx.
            # Normal approximation for large n_permutations

            z = t_stat  # treat t_stat as z-score
            p_val = 0.5 * (1 - erf(z / (sqrt(2))))  # one-sided p-value for z
            p_val = max(min(p_val, 1.0), 0.0)
            # Ensure p_val within [0,1]
            p_val = float(min(max(p_val, 0.0), 1.0))

        feature_name = feature_name_list[j] if feature_name_list is not None else j
        importance_table.append(
            {
                "feature_index": feature_name,
                "importance": importance_score,
                "95% CI": (ci_lower, ci_upper),
                "p_value": p_val,
            }
        )
    return importance_table
