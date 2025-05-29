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
    def validate_index_consistency(self) -> "MacroConfig":
        """
        Validates that the index_column_name is consistent between regression config and probe options.
        """
        regression_index = self.regression.index_column_name
        probe_index = self.probe.regression_eval_opts.index_column_name

        if regression_index != probe_index:
            raise ValueError(
                f"Inconsistent index_column_name: "
                f"regression config uses '{regression_index}' but "
                f"probe regression_eval_opts uses '{probe_index}'"
            )
        return self

    @model_validator(mode="after")
    def validate_early_stopping_regression(self) -> "MacroConfig":
        """
        Validates that regression evaluation is available when PValEarlyStoppingConfig is enabled.
        Also ensures that test set results are enabled since they are needed for early stopping validation.
        """
        if (
            isinstance(self.training.stopping_options, PValEarlyStoppingConfig)
            and self.training.stopping_options.enabled
        ):
            if not self.probe.regression_eval_opts:
                raise ValueError(
                    "PValEarlyStoppingConfig is enabled but regression_eval_opts is not set in probe options. "
                    "Regression evaluation is required for p-value based early stopping."
                )
            if not self.probe.regression_eval_opts.evaluate:
                raise ValueError(
                    "PValEarlyStoppingConfig is enabled but regression evaluation is disabled "
                    "(probe.regression_eval_opts.evaluate=False). "
                    "Regression evaluation must be enabled for p-value based early stopping."
                )
            if not self.probe.get_testset_results:
                raise ValueError(
                    "PValEarlyStoppingConfig is enabled but get_testset_results is disabled in probe options. "
                    "Test set results are required for p-value based early stopping validation."
                )
        return self
