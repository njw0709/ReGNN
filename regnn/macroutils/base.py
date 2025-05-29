from typing import Any, List, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator
from regnn.model import ReGNNConfig
from regnn.train import (
    TrainingHyperParams,
    ProbeOptions,
    KLDLossConfig,
    PValEarlyStoppingConfig,
)
from regnn.data import DataFrameReadInConfig
from regnn.probe import RegressionEvalProbeScheduleConfig, DataSource


class ModeratedRegressionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    focal_predictor: str
    outcome_col: str
    controlled_cols: List[str]
    moderators: Union[List[str], List[List[str]]]
    control_moderators: bool = Field(True)
    index_column_name: Union[str, List[str]]


class MacroConfig(BaseModel):
    """Configuration for ReGNN training"""

    model_config = ConfigDict(arbitrary_types_allowed=False, validate_assignment=True)

    # Core configurations
    read_config: DataFrameReadInConfig = Field(
        default_factory=DataFrameReadInConfig,
        description="Dataset construction configurations",
    )
    regression: ModeratedRegressionConfig = Field(
        description="regression setup configuration",
    )
    model: ReGNNConfig = Field(..., description="Model configuration")
    training: TrainingHyperParams = Field(
        default_factory=TrainingHyperParams, description="Training hyperparameters"
    )

    probe: ProbeOptions = Field(
        default_factory=ProbeOptions,
        description="While-training probe configuration",
    )

    @model_validator(mode="after")
    def check_kld_loss_compatibility(cls, data: Any) -> Any:

        # Assuming `data` is the MacroConfig instance itself after field validation:
        loss_opts = data.training.loss_options
        model_cfg = data.model

        if isinstance(loss_opts, KLDLossConfig):
            if not model_cfg.nn_config.vae:
                raise ValueError(
                    "KLDLossConfig is specified, but model.nn_config.vae is False. "
                    "The model must be a VAE (model.nn_config.vae=True) to use KLD loss."
                )
            if not model_cfg.nn_config.output_mu_var:
                raise ValueError(
                    "KLDLossConfig is specified, but model.nn_config.output_mu_var is False. "
                    "The model must output mu and logvar (model.nn_config.output_mu_var=True) for KLD loss."
                )
        return data

    @model_validator(mode="after")
    def check_survey_weights_and_loss_reduction(cls, data: Any) -> Any:
        """
        Validates compatibility between survey weight usage and loss reduction method.
        - If survey_weights are used, loss reduction must be 'none'.
        - If loss reduction is 'none', survey_weights should ideally be specified.
        """
        read_cfg = data.read_config  # DataFrameReadInConfig
        loss_opts = (
            data.training.loss_options
        )  # LossConfigs (e.g., MSELossConfig, KLDLossConfig)

        survey_weights_column = read_cfg.survey_weight_col
        loss_reduction = loss_opts.reduction

        if survey_weights_column is not None:
            # If survey weights are specified, loss reduction MUST be 'none'
            if loss_reduction != "none":
                raise ValueError(
                    f"Survey weights column '{survey_weights_column}' is specified in read_config, "
                    f"but loss reduction in training.loss_options is '{loss_reduction}'. "
                    f"Loss reduction must be 'none' when using survey weights."
                )
        else:
            # If survey weights are NOT specified, but loss reduction is 'none'
            if loss_reduction == "none":
                print(
                    f"Warning: Loss reduction is 'none' in training.loss_options, but no survey_weight_col "
                    f"is specified in read_config. Ensure this is intended if not using survey weights."
                )
        return data

    @model_validator(mode="after")
    def check_regression_vars_in_read_cols(cls, data: Any) -> Any:
        """
        Validates that all variables specified in regression_config are present in read_config.read_cols.
        This includes:
        - focal_predictor
        - outcome_col
        - controlled_cols
        - moderators
        """
        read_cols = set(data.read_config.read_cols)
        reg_config = data.regression

        # Collect all variables that should be in read_cols
        required_vars = {
            reg_config.focal_predictor,
            reg_config.outcome_col,
            *reg_config.controlled_cols,
            *reg_config.moderators,
        }

        # Find any missing variables
        missing_vars = required_vars - read_cols
        if missing_vars:
            raise ValueError(
                f"The following variables specified in regression_config are not present in read_config.read_cols: "
                f"{sorted(missing_vars)}. Please ensure all regression variables are included in read_cols."
            )

        return data

    @model_validator(mode="after")
    def validate_early_stopping_probes(self) -> "MacroConfig":
        """
        If PValEarlyStoppingConfig is enabled, validates that:
        1. RegressionEvalProbeScheduleConfig is scheduled for DataSource.TRAIN.
        2. RegressionEvalProbeScheduleConfig is scheduled for DataSource.TEST.
        These are needed to get p-values for early stopping.
        """
        if not (
            isinstance(self.training.stopping_options, PValEarlyStoppingConfig)
            and self.training.stopping_options.enabled
        ):
            return self  # Early stopping not enabled or not PVal type

        has_train_pval_probe = False
        has_test_pval_probe = False

        for schedule in self.probe.schedules:
            if isinstance(schedule, RegressionEvalProbeScheduleConfig):
                # Check if it runs on TRAIN data
                if DataSource.TRAIN in schedule.data_sources:
                    has_train_pval_probe = True

                # Check if it runs on TEST data
                if DataSource.TEST in schedule.data_sources:
                    has_test_pval_probe = True

            if has_train_pval_probe and has_test_pval_probe:
                break  # Both found

        if not has_train_pval_probe:
            raise ValueError(
                "PValEarlyStoppingConfig is enabled, but no 'regression_eval' probe is scheduled "
                "to run on DataSource.TRAIN. This is required for p-value based early stopping."
            )

        if not has_test_pval_probe:
            raise ValueError(
                "PValEarlyStoppingConfig is enabled, but no 'regression_eval' probe is scheduled "
                "to run on DataSource.TEST. This is required for p-value based early stopping."
            )

        return self

    @model_validator(mode="after")
    def validate_probe_schedule_consistency(self) -> "MacroConfig":
        """
        Validates specific probe schedule configurations for consistency:
        - For RegressionEvalProbeScheduleConfig: checks if its index_column_name
          matches the main regression.index_column_name.
        """
        main_regression_index = self.regression.index_column_name

        for schedule in self.probe.schedules:
            if isinstance(schedule, RegressionEvalProbeScheduleConfig):
                if schedule.index_column_name != main_regression_index:
                    raise ValueError(
                        f"Inconsistent index_column_name for a 'regression_eval' probe: "
                        f"Probe schedule uses '{schedule.index_column_name}', but "
                        f"main regression config uses '{main_regression_index}'. "
                        f"These must match for 'regression_eval' probes."
                    )
        return self
