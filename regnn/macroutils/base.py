from typing import Any, List, Union, Set
from pydantic import BaseModel, Field, ConfigDict, model_validator
from regnn.model import ReGNNConfig
from regnn.train import (
    TrainingHyperParams,
    KLDLossConfig,
)
from regnn.data import DataFrameReadInConfig
from regnn.probe import (
    RegressionEvalProbeScheduleConfig,
    DataSource,
    PValEarlyStoppingProbeScheduleConfig,
    FrequencyType,
    ProbeOptions,
)
import warnings


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
        loss_opts = data.training.loss_options
        model_cfg = data.model
        if isinstance(loss_opts, KLDLossConfig):
            if not model_cfg.nn_config.vae:
                raise ValueError(
                    "KLDLossConfig specified, but model.nn_config.vae is False."
                )
            if not model_cfg.nn_config.output_mu_var:
                raise ValueError(
                    "KLDLossConfig specified, but model.nn_config.output_mu_var is False."
                )
        return data

    @model_validator(mode="after")
    def check_survey_weights_and_loss_reduction(cls, data: Any) -> Any:
        """
        Validates compatibility between survey weight usage and loss reduction method.
        - If survey_weights are used, loss reduction must be 'none'.
        - If loss reduction is 'none', survey_weights should ideally be specified.
        """
        read_cfg = data.read_config
        loss_opts = data.training.loss_options
        survey_weights_column = read_cfg.survey_weight_col
        loss_reduction = loss_opts.reduction
        print(loss_reduction)

        if survey_weights_column is not None:
            # If survey weights are specified, loss reduction MUST be 'none'
            if loss_reduction != "none":
                raise ValueError(
                    f"Survey weights column '{survey_weights_column}' used, but loss reduction is '{loss_reduction}'. Must be 'none'."
                )
        else:
            # If survey weights are NOT specified, but loss reduction is 'none'
            if loss_reduction == "none":
                warnings.warn(
                    "Loss reduction is 'none', but no survey_weight_col specified. Ensure intended.",
                    UserWarning,
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
        }

        if reg_config.moderators:
            if isinstance(reg_config.moderators[0], list):
                for mod_group in reg_config.moderators:
                    required_vars.update(mod_group)
            else:
                required_vars.update(reg_config.moderators)

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
        If a PValEarlyStoppingProbeScheduleConfig is present, validates that for each of its
        monitored data sources, a corresponding RegressionEvalProbeScheduleConfig is also scheduled.
        """
        early_stopping_schedules = [
            s
            for s in self.probe.schedules
            if isinstance(s, PValEarlyStoppingProbeScheduleConfig)
        ]

        if not early_stopping_schedules:
            return self  # No p-value early stopping probe scheduled, nothing to validate here.

        # Get all data sources for which RegressionEvalProbes are scheduled
        regression_eval_covered_sources: Set[DataSource] = set()
        for sched in self.probe.schedules:
            if isinstance(sched, RegressionEvalProbeScheduleConfig):
                for ds in sched.data_sources:
                    regression_eval_covered_sources.add(ds)

        for es_schedule in early_stopping_schedules:
            for monitored_ds in es_schedule.data_sources_to_monitor:
                found_matching_reg_eval = False
                is_reg_eval_frequent_enough = False
                for reg_eval_schedule in self.probe.schedules:
                    if (
                        isinstance(reg_eval_schedule, RegressionEvalProbeScheduleConfig)
                        and monitored_ds in reg_eval_schedule.data_sources
                    ):
                        found_matching_reg_eval = True
                        # Check frequency only if the early stopping probe itself is frequent
                        if (
                            es_schedule.frequency_type == FrequencyType.EPOCH
                            and es_schedule.frequency_value == 1
                        ):
                            if (
                                reg_eval_schedule.frequency_type == FrequencyType.EPOCH
                                and reg_eval_schedule.frequency_value == 1
                            ):
                                is_reg_eval_frequent_enough = True
                            else:
                                # Matching reg eval found, but not frequent enough for an aggressive early stopper
                                is_reg_eval_frequent_enough = (
                                    False  # Explicitly false, warning will be issued
                                )
                                break  # Found a reg_eval for this ds, no need to check others for this ds
                        else:
                            # Early stopper is not aggressive, so any reg_eval frequency is fine for existence
                            is_reg_eval_frequent_enough = (
                                True  # Mark as fine from frequency perspective
                            )
                        break  # Found a reg_eval for this ds

                if not found_matching_reg_eval:
                    raise ValueError(
                        f"PValEarlyStoppingProbe is configured to monitor DataSource '{monitored_ds}', "
                        f"but no 'regression_eval' probe is scheduled to run on this data source. "
                        f"P-value based early stopping requires regression evaluations on all monitored sources."
                    )

                # Issue warning if early stopping is aggressive but regression eval is not
                if (
                    es_schedule.frequency_type == FrequencyType.EPOCH
                    and es_schedule.frequency_value == 1
                    and not is_reg_eval_frequent_enough
                    and found_matching_reg_eval
                ):
                    warnings.warn(
                        f"PValEarlyStoppingProbe is scheduled to check DataSource '{monitored_ds}' every epoch, "
                        f"but the corresponding 'regression_eval' probe for this data source is NOT scheduled every epoch. "
                        f"This may lead to early stopping decisions based on stale p-values. "
                        f"Consider aligning the 'regression_eval' probe's frequency_value to 1 for '{monitored_ds}' for more reactive early stopping.",
                        UserWarning,
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
                if isinstance(main_regression_index, list):
                    # This case needs clarification: if main index can be a list, how does it map to a single probe index_column_name?
                    # For now, if main is list, we might skip this check or require probe index to be in the list.
                    # Or, perhaps RegressionEvalProbeScheduleConfig.index_column_name should also be able to be a list?
                    # Current logic implies both are str.
                    pass
                elif schedule.index_column_name != main_regression_index:
                    raise ValueError(
                        f"Inconsistent index_column_name for 'regression_eval' probe: "
                        f"Uses '{schedule.index_column_name}', main is '{main_regression_index}'. Must match."
                    )
        return self
