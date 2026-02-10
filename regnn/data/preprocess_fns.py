import pandas as pd
import numpy as np
from typing import Sequence, Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from .debias import debias_treatment_kfold


# preprocess categorical col to one hot encoding
def binary_to_one_hot(
    df: pd.DataFrame, binary_cats: Optional[Sequence[str]] = None, dtype: str = "float"
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if binary_cats is None:
        return df
    for bc in binary_cats:
        if df[bc].dtype != "category":
            col = df[bc].astype("category")
        else:
            col = df[bc]
        df[bc] = col.cat.codes
    return df, binary_cats


def _reverse_binary_to_one_hot(
    df: pd.DataFrame, binary_cats: Optional[Sequence[str]] = None, **kwargs
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if binary_cats is None:
        return df
    for bc in binary_cats:
        # Since we used cat.codes, we need to map back to original categories
        # This assumes the original categories were 0 and 1
        if bc in df.columns:
            df[bc] = df[bc].map({0: False, 1: True}).astype("category")
    return df, binary_cats


binary_to_one_hot._reverse_transform = _reverse_binary_to_one_hot
binary_to_one_hot.__name__ = "binary_to_one_hot"


def multi_cat_to_one_hot(
    df: pd.DataFrame, multi_cats: Optional[Sequence[str]] = None, dtype: str = "float"
) -> Tuple[pd.DataFrame, Dict[str, Sequence[str]]]:
    if multi_cats is None or len(multi_cats) == 0:
        return df, {}

    # Store original categories for each column before transformation
    categories_dict = {col: df[col].cat.categories for col in multi_cats}
    # Get dummies with drop_first=True
    df2 = pd.get_dummies(
        df[multi_cats], columns=multi_cats, dtype=float, drop_first=True
    )

    if dtype == "category":
        for c in df2.columns:
            df2[c] = df2[c].astype(dtype)
    df = pd.concat([df, df2], axis=1)
    df.drop(multi_cats, inplace=True, axis=1)
    return_dict = {"category_map": categories_dict}
    return df, return_dict


def _reverse_multi_cat_to_one_hot(
    df: pd.DataFrame,
    multi_cats: Optional[Sequence[str]] = None,
    category_map: Dict[str, Sequence[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if multi_cats is None:
        return df
    if category_map is None:
        raise ValueError("Category mapping dictionary must be provided!!")

    # Reconstruct original categories from one-hot encoded columns
    for cat in multi_cats:
        if cat in category_map:
            original_categories = category_map[cat]
            # Get one-hot column names from original categories (excluding first category)
            cat_cols = [f"{cat}_{cat_val}" for cat_val in original_categories[1:]]

            # Initialize with first category (the one that was dropped)
            first_category = original_categories[0]
            values = [first_category] * len(df)

            # Now assign other categories based on one-hot columns
            for i, category in enumerate(original_categories[1:], 1):
                col = f"{cat}_{category}"
                if col in df.columns:  # Check if column exists
                    mask = df[col] == 1
                    # Only update values where mask is True
                    for idx in mask[mask].index:
                        values[idx] = category

            # Create categorical column with the correct categories
            df[cat] = pd.Categorical(values, categories=original_categories)

            # Drop the one-hot encoded columns
            df.drop(columns=cat_cols, inplace=True, errors="ignore")

    return df, multi_cats


multi_cat_to_one_hot._reverse_transform = _reverse_multi_cat_to_one_hot
multi_cat_to_one_hot.__name__ = "multi_cat_to_one_hot"


def standardize_cols(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    if columns is None:
        columns = df.columns
    mean_std_dict = {}
    for c in columns:
        if df[c].dtype != "category":
            mean = df[c].mean()
            std = df[c].std()
            df[c] = (df[c] - mean) / std
            mean_std_dict[c] = (mean, std)
        else:
            print("is category: ", c)
    return_dict = {"mean_std_dict": mean_std_dict}
    return df, return_dict


def _reverse_standardize_cols(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    mean_std_dict: Dict[str, Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if columns is None:
        return df
    if mean_std_dict is None:
        raise ValueError(
            "mean std dictionary must be provided to reverse standardization!!!"
        )
    for c in columns:
        if c in mean_std_dict and c in df.columns and df[c].dtype != "category":
            mean, std = mean_std_dict[c]
            df[c] = (df[c] * std) + mean
    return df, columns


standardize_cols._reverse_transform = _reverse_standardize_cols
standardize_cols.__name__ = "standardize_cols"


def convert_categorical_to_ordinal(
    df: pd.DataFrame, ordinal_cols: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if ordinal_cols is None:
        ordinal_cols = df.columns
    for c in ordinal_cols:
        if df[c].dtype == "float":
            continue
        elif df[c].dtype == "category":
            df[c] = df[c].cat.codes
            df[c] = df[c].astype("float")
    return df, ordinal_cols


def _reverse_convert_categorical_to_ordinal(
    df: pd.DataFrame, ordinal_cols: Optional[Sequence[str]] = None, **kwargs
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if ordinal_cols is None:
        return df
    for c in ordinal_cols:
        if c in df.columns:
            # Convert back to categorical type
            df[c] = df[c].astype("category")
    return df, ordinal_cols


convert_categorical_to_ordinal._reverse_transform = (
    _reverse_convert_categorical_to_ordinal
)
convert_categorical_to_ordinal.__name__ = "convert_categorical_to_ordinal"


def map_to_zero_one(
    df: pd.DataFrame, cols: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if cols is None:
        cols = df.columns
    min_max_dict = {}
    for c in cols:
        col_min = df[c].min()
        col_max = df[c].max()
        df[c] = (df[c] - col_min) / (col_max - col_min)
        min_max_dict[c] = (col_min, col_max)
    return_dict = {"min_max_dict": min_max_dict}
    return df, return_dict


def _reverse_map_to_zero_one(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    min_max_dict: Dict[str, Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if cols is None:
        cols = df.columns
    if min_max_dict is None:
        min_max_dict = {}
    for c in cols:
        if c in min_max_dict and c in df.columns:
            col_min, col_max = min_max_dict[c]
            df[c] = (df[c] * (col_max - col_min)) + col_min
    return df, min_max_dict


map_to_zero_one._reverse_transform = _reverse_map_to_zero_one
map_to_zero_one.__name__ = "map_to_zero_one"


def debias_focal_predictor(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    controlled_predictors: Optional[Sequence[str]] = None,
    model_class: Optional[type] = None,
    k: int = 5,
    is_classifier: bool = False,
    sample_weight_col: Optional[str] = None,
    **model_params,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Debias the focal predictor using k-fold cross-validation.

    This function removes the confounding effect of controlled predictors from
    the focal predictor by fitting models to predict the focal predictor from
    the controlled predictors, then computing residuals.

    Args:
        df: DataFrame containing the data
        columns: List containing the focal predictor column name (should be length 1)
        controlled_predictors: List of controlled predictor column names to use as covariates
        model_class: The class of the model (e.g., LogisticRegression, Ridge).
                    If None, defaults to RandomForestRegressor or RandomForestClassifier
        k: Number of folds for cross-validation
        is_classifier: True for binary focal predictor (uses propensity scores),
                      False for continuous focal predictor
        sample_weight_col: Column name containing sample weights for model fitting.
                          If None, all samples are weighted equally.
        **model_params: Additional parameters to pass to the model

    Returns:
        Tuple of (modified DataFrame, metadata dict)
        The metadata dict contains:
            - models: List of k trained models
            - predictions: Array of predictions (propensity scores or E[D|X])
            - original_values: Original focal predictor values
            - controlled_predictors: List of controlled predictor column names
    """
    if columns is None or len(columns) == 0:
        raise ValueError("Must specify focal predictor column in 'columns' parameter")
    if len(columns) > 1:
        raise ValueError("Can only debias one focal predictor at a time")
    if controlled_predictors is None or len(controlled_predictors) == 0:
        raise ValueError("Must specify controlled predictors")

    focal_col = columns[0]

    # Extract focal predictor and controlled predictors
    D = df[focal_col].values
    X = df[controlled_predictors].values
    
    # Extract sample weights if provided
    sample_weight = df[sample_weight_col].values if sample_weight_col is not None else None

    # Set default model if not provided
    if model_class is None:
        if is_classifier:
            model_class = LogisticRegression
            if not model_params:
                model_params = {"penalty": "l1", "solver": "liblinear", "random_state": 42}
        else:
            model_class = Lasso
            if not model_params:
                model_params = {"alpha": 1.0, "random_state": 42}

    # Call debias_treatment_kfold
    D_tilde, predictions, models = debias_treatment_kfold(
        X, D, model_class, k=k, is_classifier=is_classifier, sample_weight=sample_weight, **model_params
    )

    # Replace focal predictor column with residuals
    df[focal_col] = D_tilde

    # Return metadata for reverse transformation
    return_dict = {
        "models": models,
        "predictions": predictions,
        "original_values": D.copy(),
        "controlled_predictors": list(controlled_predictors),
    }

    return df, return_dict


def _reverse_debias_focal_predictor(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    models: Optional[List] = None,
    predictions: Optional[np.ndarray] = None,
    original_values: Optional[np.ndarray] = None,
    controlled_predictors: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Reverse the debiasing transformation by adding predictions back to residuals.

    Args:
        df: DataFrame with debiased focal predictor
        columns: List containing the focal predictor column name
        models: List of trained models (not used in basic reverse, but stored for reference)
        predictions: Array of predictions from the debiasing step
        original_values: Original focal predictor values (fallback if predictions not available)
        controlled_predictors: List of controlled predictor column names
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Tuple of (modified DataFrame, column names)
    """
    if columns is None or len(columns) == 0:
        raise ValueError("Must specify focal predictor column in 'columns' parameter")

    focal_col = columns[0]

    # Use original values if available (most accurate)
    if original_values is not None:
        df[focal_col] = original_values
    elif predictions is not None:
        # Reverse: D_original = D_tilde + predictions
        df[focal_col] = df[focal_col] + predictions
    else:
        raise ValueError(
            "Either 'original_values' or 'predictions' must be provided for reverse transformation"
        )

    return df, columns


debias_focal_predictor._reverse_transform = _reverse_debias_focal_predictor
debias_focal_predictor.__name__ = "debias_focal_predictor"
