import pandas as pd
from regnn.data.process import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
    map_to_zero_one,
)
from regnn.data.dataset import ReGNNDataset
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
    df_dtype_list = [("category", c) for c in categorical_cols]
    df_dtype_list += [("ordinals", c) for c in ordinal_cols]
    df_dtype_list += [("binary", c) for c in binary_cols]
    df_dtype_list += [("continuous", c) for c in continuous_cols]
    df_dtypes = dict(df_dtype_list)

    # make ReGNN dataset
    regnn_dataset = ReGNNDataset(
        df,
        focal_predictor,
        controlled_cols,
        moderators,
        outcome_col,
        survey_weights,
        rename_dict=rename_dict,
        df_dtypes=df_dtypes,
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

    regnn_dataset.preprocess(preprocess_list, inplace=True)
    regnn_dataset.standardize(standardize_list, inplace=True)
    regnn_dataset.dropna(inplace=True)

    return regnn_dataset
