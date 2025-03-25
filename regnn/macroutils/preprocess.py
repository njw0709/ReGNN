import pandas as pd
from ..data.process import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
    map_to_zero_one,
)
from ..data.dataset import MIHMDataset
from typing import List, Sequence, Tuple, Union


def preprocess(
    data_path: str,
    read_cols: Sequence[str],
    rename_dict: dict,
    binary_cols: List[str],
    categorical_cols: List[str],
    ordinal_cols: List[str],
    continuous_cols: List[str],
    focal_predictor: str,
    outcome_col: str,
    controlled_cols: List[str],
    moderators: List[str],
    survey_weights: Union[str, None] = None,
):
    # read data
    df = pd.read_stata(data_path, columns=read_cols)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_orig = df.copy()
    df.rename(columns=rename_dict, inplace=True)

    # get binary and multi category columns
    for c in categorical_cols:
        df[c] = df[c].astype("category")

    for c in binary_cols:
        df[c] = df[c].astype("category")

    # make MIHM dataset
    heat_dataset = MIHMDataset(
        df, focal_predictor, controlled_cols, moderators, outcome_col, survey_weights
    )

    # preprocess data
    preprocess_list = [
        (binary_cols, binary_to_one_hot),
        (categorical_cols, multi_cat_to_one_hot),
        (ordinal_cols, convert_categorical_to_ordinal),
    ]
    if survey_weights:
        preprocess_list.append(([survey_weights], map_to_zero_one))

    standardize_list = [(continuous_cols + ordinal_cols, standardize_cols)]

    heat_dataset.preprocess(preprocess_list, inplace=True)
    heat_dataset.standardize(standardize_list, inplace=True)
    heat_dataset.dropna(inplace=True)

    return df_orig, heat_dataset
