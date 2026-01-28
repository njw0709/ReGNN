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
    preprocess_steps = list(read_config.preprocess_steps)  # Make a copy
    
    # Add debiasing step if enabled
    if regression_config.debias_treatment:
        from regnn.data.preprocess_fns import debias_focal_predictor
        from regnn.data.base import PreprocessStep
        from regnn.probe.dataclass.regression import DebiasConfig
        
        # Get or create debias config
        debias_cfg = regression_config.debias_config
        if debias_cfg is None:
            debias_cfg = DebiasConfig()
        
        # Get all predictors for debiasing (controlled + moderators)
        controlled_for_debias = regression_config.controlled_cols.copy()
        if isinstance(regression_config.moderators[0], list):
            for mod_group in regression_config.moderators:
                controlled_for_debias.extend(mod_group)
        else:
            controlled_for_debias.extend(regression_config.moderators)
        
        # Create debias step
        debias_step = PreprocessStep(
            columns=[regression_config.focal_predictor],
            function=debias_focal_predictor,
        )
        # Store additional parameters in reverse_transform_info for use during preprocessing
        # Note: controlled_predictors will be updated dynamically during preprocessing
        # to reflect transformed column names (e.g., after one-hot encoding)
        debias_step.reverse_transform_info = {
            'controlled_predictors': controlled_for_debias,
            'model_class': debias_cfg.model_class,
            'k': debias_cfg.k,
            'is_classifier': debias_cfg.is_classifier,
            **debias_cfg.model_params
        }
        
        # Append at the END of preprocessing pipeline
        # The controlled_predictors list will be updated during preprocessing
        # to reflect any column transformations (one-hot encoding, etc.)
        preprocess_steps.append(debias_step)

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
