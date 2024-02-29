import pandas as pd
from ..data.process import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
)
from ..data.dataset import MIHMDataset
from typing import List, Sequence, Tuple, Union


def preprocess(
    data_path: str,
    read_cols: Sequence[str],
    rename_dict: dict,
    categorical_cols: List[str],
    ordinal_cols: List[str],
    continuous_cols: List[str],
    interactor_col: str,
    outcome_col: str,
    controlled_cols: List[str],
    interaction_predictors: List[str],
):
    # read data
    df = pd.read_stata(data_path, columns=read_cols)
    df_orig = df.copy()
    df_orig.dropna(inplace=True)
    df.rename(columns=rename_dict, inplace=True)

    # get binary and multi category columns
    for c in categorical_cols:
        df[c] = df[c].astype("category")
    binary_cats = [c for c in categorical_cols if df[c].nunique() <= 2]
    multi_cats = [c for c in categorical_cols if df[c].nunique() > 2]

    # make MIHM dataset
    heat_dataset = MIHMDataset(
        df, interactor_col, controlled_cols, interaction_predictors, outcome_col
    )

    # preprocess data
    preprocess_list = [
        (binary_cats, binary_to_one_hot),
        (multi_cats, multi_cat_to_one_hot),
        (ordinal_cols, convert_categorical_to_ordinal),
    ]

    standardize_list = [(continuous_cols + ordinal_cols, standardize_cols)]

    heat_dataset.preprocess(preprocess_list, inplace=True)
    heat_dataset.standardize(standardize_list, inplace=True)
    heat_dataset.dropna(inplace=True)

    return df_orig, heat_dataset

