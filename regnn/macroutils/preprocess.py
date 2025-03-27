import pandas as pd
from regnn.data.preprocess_fns import (
    multi_cat_to_one_hot,
    binary_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
    map_to_zero_one,
)
from regnn.data.dataset import ReGNNDataset
from regnn.data.base import ReGNNDatasetConfig, PreprocessStep
from typing import List, Sequence, Union


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

    # Create preprocessing steps
    preprocess_steps = [
        PreprocessStep(columns=binary_cols, function=binary_to_one_hot),
        PreprocessStep(columns=categorical_cols, function=multi_cat_to_one_hot),
        PreprocessStep(columns=ordinal_cols, function=convert_categorical_to_ordinal),
        PreprocessStep(
            columns=continuous_cols + ordinal_cols, function=standardize_cols
        ),
    ]

    if survey_weights:
        preprocess_steps.append(
            PreprocessStep(columns=[survey_weights], function=map_to_zero_one)
        )

    # Create config with preprocessing steps
    config = ReGNNDatasetConfig(
        focal_predictor=focal_predictor,
        controlled_predictors=controlled_cols,
        moderators=moderators,
        outcome=outcome_col,
        survey_weights=survey_weights,
        rename_dict=rename_dict,
        df_dtypes=df_dtypes,
        preprocess_steps=preprocess_steps,
    )

    # make ReGNN dataset
    regnn_dataset = ReGNNDataset(
        df,
        config=config,
    )

    # preprocess data
    regnn_dataset.preprocess(inplace=True)
    regnn_dataset.dropna(inplace=True)

    return regnn_dataset
