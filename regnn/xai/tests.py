import pandas as pd
from scipy.stats import chisquare
import numpy as np
from typing import Tuple


def categorical_test(
    cluster_col: np.ndarray, population_col: np.ndarray
) -> Tuple[float, float]:
    """Perform chi-square test comparing categorical distributions.

    Args:
        cluster_col: Array of categorical values for the cluster
        population_col: Array of categorical values for the population

    Returns:
        Tuple[float, float]: (chi-square statistic, p-value)

    Raises:
        ValueError: If inputs are invalid or assumptions are violated
    """
    if cluster_col.ndim != 1 or population_col.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if len(cluster_col) == 0 or len(population_col) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Convert to pandas Series and handle NaN values consistently
    cluster_series = pd.Series(cluster_col).astype("category")
    population_series = pd.Series(population_col).astype("category")

    # Get all categories from both series
    all_categories = pd.Index(
        pd.concat([cluster_series, population_series]).dropna().unique()
    ).sort_values()

    if len(all_categories) == 0:
        raise ValueError("No valid categories found after removing NaN values")

    # Get counts and proportions
    cluster_counts = (
        cluster_series.value_counts().reindex(all_categories, fill_value=0).sort_index()
    )
    population_proportions = (
        population_series.value_counts(normalize=True)
        .reindex(all_categories, fill_value=0)
        .sort_index()
    )

    # Calculate expected counts
    expected_counts = population_proportions * len(cluster_series)

    # Check chi-square assumptions
    if (expected_counts < 5).any():
        raise ValueError(
            "Chi-square test assumption violated: all expected counts should be >= 5. "
            "Consider combining rare categories or using a different test."
        )

    # Check for zero expected counts
    if (expected_counts == 0).any():
        raise ValueError(
            "Zero expected counts found. This can happen when a category exists in "
            "the cluster but not in the population."
        )

    # Perform chi-square test
    try:
        chi2, p_val = chisquare(
            f_obs=cluster_counts.values, f_exp=expected_counts.values
        )
        return chi2, p_val
    except Exception as e:
        raise ValueError(f"Error in chi-square test: {str(e)}")
