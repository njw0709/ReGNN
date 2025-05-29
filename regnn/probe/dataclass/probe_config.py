from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, field_validator, ConfigDict, Field

from regnn.eval import FocalPredictorPreProcessOptions

# Forward declaration for ProbeData if it's defined elsewhere and causes circular imports
# For now, we assume ProbeData will be a base class or a well-defined type.
# from regnn.probe import ProbeData # This might be needed later


class FrequencyType(str, Enum):
    """Defines how frequently a probe should be run."""

    PRE_TRAINING = "pre_training"  # Once before the training loop starts
    POST_TRAINING = "post_training"  # Once after the training loop finishes
    EPOCH = "epoch"  # At the end of specified epochs
    ITERATION = "iteration"  # After specified iterations (batches)


class DataSource(str, Enum):
    """Specifies the data source a probe should operate on."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"  # If a dedicated validation set is used
    ALL = "all"  # For probes that operate on the model globally or combined data


class ProbeScheduleConfig(BaseModel):
    """Configuration for scheduling a specific probe."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    probe_type: str  # String identifier for the probe (e.g., "save_model", "l2_norm")
    frequency_type: FrequencyType
    frequency_value: int = (
        1  # e.g., every 1 epoch, every 10 iterations. For PRE/POST, this is ignored.
    )
    data_sources: List[DataSource] = [
        DataSource.TRAIN
    ]  # Default to train if not specified
    probe_params: Optional[Dict[str, Any]] = (
        None  # Specific parameters for this probe instance
    )

    @field_validator("frequency_value")
    def check_frequency_value(cls, v, info):
        # Pydantic v2: info is an instance of ValidationInfo
        # Access field values via info.data, which is a dict of the model's fields
        if info.data.get("frequency_type") in [
            FrequencyType.EPOCH,
            FrequencyType.ITERATION,
        ]:
            if v < 1:
                raise ValueError(
                    "frequency_value must be at least 1 for EPOCH or ITERATION frequency types."
                )
        return v

    @field_validator("data_sources", mode="before")
    def ensure_data_sources_is_list(cls, v):
        if isinstance(v, (str, DataSource)):
            return [v]
        if not v:  # Handle None or empty by defaulting
            return [DataSource.TRAIN]
        return v


class RegressionEvalProbeScheduleConfig(ProbeScheduleConfig):
    """Configuration for a regression evaluation probe."""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    probe_type: Literal["regression_eval"] = "regression_eval"

    # Fields from former RegressionEvalOptions
    regress_cmd: str = Field(
        ...,
        description="Regression command to run, e.g., 'dep_var ~ indep_var1 + indep_var2'",
    )
    evaluation_function: Literal["stata", "statsmodels"] = Field(
        "statsmodels", description="Evaluation engine to use"
    )
    focal_predictor_process_options: FocalPredictorPreProcessOptions = Field(
        default_factory=FocalPredictorPreProcessOptions,
        description="Options for preprocessing the focal predictor before evaluation",
    )
    index_column_name: str = Field(
        "summary_index",
        description="Name of the index column if used in regression or for identification",
    )
    # 'evaluate' and 'eval_epochs' are covered by ProbeScheduleConfig's frequency_type/value
    # 'post_training_eval' is covered by setting frequency_type = FrequencyType.POST_TRAINING

    @field_validator("regress_cmd")
    def validate_regress_cmd(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("regress_cmd cannot be empty")
        # Basic check, can be enhanced
        if (
            "~" not in v and len(v.split()) < 3
        ):  # Support for patsy formula or space separated
            raise ValueError(
                "regress_cmd should be a valid regression formula (e.g., 'y ~ x1 + x2') or "
                "a command with at least 3 parts: command, dependent var, and independent var for older formats."
            )
        return v


class SaveCheckpointProbeScheduleConfig(ProbeScheduleConfig):
    """Configuration for saving model checkpoints."""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    probe_type: Literal["save_checkpoint"] = "save_checkpoint"

    save_dir: str = Field(..., description="Directory to save model checkpoints.")
    model_save_name: str = Field(
        "regnn_model", description="Base name for saved model files."
    )
    file_id: Optional[str] = Field(
        None, description="Optional suffix for filenames, e.g., an experiment ID."
    )
    # epoch_in_filename: bool = Field(True, description="Whether to include epoch number in the filename.")


class SaveIntermediateIndexProbeScheduleConfig(ProbeScheduleConfig):
    """Configuration for saving intermediate index predictions."""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    probe_type: Literal["save_intermediate_index"] = "save_intermediate_index"

    save_dir: str = Field(
        ..., description="Directory to save intermediate index files."
    )
    model_save_name: str = Field(
        "regnn_model",
        description="Base name used for constructing the index filename, usually matching model checkpoints.",
    )
    file_id: Optional[str] = Field(
        None,
        description="Optional suffix for filenames, often an experiment ID to make filenames unique like '{model_save_name}-{file_id}-indices.dta'.",
    )


class GetObjectiveProbeScheduleConfig(ProbeScheduleConfig):
    """
    Configuration for a probe that calculates accuracy or other relevant metrics.
    The actual metric calculated would depend on the probe's implementation.
    """

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    probe_type: Literal["objective"] = "objective"
    # Example parameters:
    # metric_name: str = Field("accuracy", description="Name of the primary metric to compute.")
    # additional_metrics: List[str] = Field(default_factory=list, description="Other metrics to compute.")


class GetL2LengthProbeScheduleConfig(ProbeScheduleConfig):
    """Configuration for calculating L2 norm of model parameters."""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    probe_type: Literal["l2_length"] = "l2_length"
    # This probe typically doesn't need extra parameters beyond frequency and data_sources (usually ALL or none).
    # It operates directly on the model.
    # Ensure data_sources defaults appropriately or is set for this type if needed (e.g., often [DataSource.ALL])
