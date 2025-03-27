import pandas as pd
import numpy as np
from typing import Sequence, Dict, List, Tuple, Any, Optional


# preprocess categorical col to one hot encoding
def binary_to_one_hot(
    df: pd.DataFrame, binary_cats: Optional[Sequence[str]] = None, dtype: str = "float"
) -> Tuple[pd.DataFrame, Sequence[str]]:
    one_hot_maps = {}
    if binary_cats is None:
        binary_cats = df.columns
    for bc in binary_cats:
        df[bc] = df[bc].cat.codes
    return df, binary_cats


def _reverse_binary_to_one_hot(
    df: pd.DataFrame, binary_cats: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if binary_cats is None:
        binary_cats = df.columns
    for bc in binary_cats:
        # Since we used cat.codes, we need to map back to original categories
        # This assumes the original categories were 0 and 1
        df[bc] = df[bc].map({0: False, 1: True}).astype("category")
    return df, binary_cats


binary_to_one_hot._reverse_transform = _reverse_binary_to_one_hot


def multi_cat_to_one_hot(
    df: pd.DataFrame, multi_cats: Optional[Sequence[str]] = None, dtype: str = "float"
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if multi_cats is None:
        multi_cats = df.columns
    df2 = pd.get_dummies(
        df[multi_cats], columns=multi_cats, dtype=float, drop_first=True
    )
    new_colnames = df2.columns
    if dtype == "category":
        for c in df2.columns:
            df2[c] = df2[c].astype(dtype)
    df = pd.concat([df, df2], axis=1)
    df.drop(multi_cats, inplace=True, axis=1)
    return df, new_colnames


def _reverse_multi_cat_to_one_hot(
    df: pd.DataFrame,
    multi_cats: Optional[Sequence[str]] = None,
    column_mapping: Dict[str, List[str]] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if multi_cats is None:
        multi_cats = df.columns
    if column_mapping is None:
        column_mapping = {}
    # Reconstruct original categories from one-hot encoded columns
    for cat in multi_cats:
        if cat in column_mapping:
            cat_cols = column_mapping[cat]
            if cat_cols:
                # Get the first category as the reference category
                ref_cat = cat_cols[0].split("_", 1)[1]
                # Create a mapping from one-hot columns to categories
                cat_map = {col: col.split("_", 1)[1] for col in cat_cols}
                # Create a new column with the original categories
                df[cat] = pd.Series(index=df.index, dtype="category")
                for col in cat_cols:
                    mask = df[col] == 1
                    df.loc[mask, cat] = cat_map[col]
                # Fill remaining values with the reference category
                df[cat] = df[cat].fillna(ref_cat)
                # Drop the one-hot encoded columns
                df.drop(columns=cat_cols, inplace=True)
    return df, multi_cats


multi_cat_to_one_hot._reverse_transform = _reverse_multi_cat_to_one_hot


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
    return df, mean_std_dict


def _reverse_standardize_cols(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    mean_std_dict: Dict[str, Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if columns is None:
        columns = df.columns
    if mean_std_dict is None:
        mean_std_dict = {}
    for c in columns:
        if c in mean_std_dict and df[c].dtype != "category":
            mean, std = mean_std_dict[c]
            df[c] = (df[c] * std) + mean
    return df, columns


standardize_cols._reverse_transform = _reverse_standardize_cols


def convert_categorical_to_ordinal(
    df: pd.DataFrame, ordinal_cols: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if ordinal_cols is None:
        ordinal_cols = df.columns
    for c in ordinal_cols:
        df[c] = df[c].cat.codes
        df[c] = df[c].astype("float")
    return df, ordinal_cols


def _reverse_convert_categorical_to_ordinal(
    df: pd.DataFrame, ordinal_cols: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if ordinal_cols is None:
        ordinal_cols = df.columns
    for c in ordinal_cols:
        # Convert back to categorical type
        df[c] = df[c].astype("category")
    return df, ordinal_cols


convert_categorical_to_ordinal._reverse_transform = (
    _reverse_convert_categorical_to_ordinal
)


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
    return df, cols


def _reverse_map_to_zero_one(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    min_max_dict: Dict[str, Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    if cols is None:
        cols = df.columns
    if min_max_dict is None:
        min_max_dict = {}
    for c in cols:
        if c in min_max_dict:
            col_min, col_max = min_max_dict[c]
            df[c] = (df[c] * (col_max - col_min)) + col_min
    return df, cols


map_to_zero_one._reverse_transform = _reverse_map_to_zero_one
