from regnn.data import ReGNNDataset, ReGNNDatasetConfig, DataFrameReadInConfig
from regnn.probe import ModeratedRegressionConfig


def read_and_preprocess(
    read_config: DataFrameReadInConfig,
    regression_config: ModeratedRegressionConfig,
) -> ReGNNDataset:
    """Preprocess data using DataFrameReadInConfig and create ReGNN dataset.

    Args:
        read_config: Configuration for reading and preprocessing the dataframe
        regression_config: Configuration for the moderated regression model

    Returns:
        ReGNNDataset: Preprocessed dataset ready for ReGNN
    """
    # Get dataframe and preprocessing steps from config
    df = read_config.read_df()
    df_dtypes = read_config.df_dtypes
    preprocess_steps = read_config.preprocess_steps

    # Create config with preprocessing steps
    config = ReGNNDatasetConfig(
        focal_predictor=regression_config.focal_predictor,
        controlled_predictors=regression_config.controlled_cols,
        moderators=regression_config.moderators,
        outcome=regression_config.outcome_col,
        survey_weights=read_config.survey_weight_col,
        rename_dict=read_config.rename_dict,
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
