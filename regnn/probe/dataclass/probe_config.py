from enum import Enum
from typing import List, Optional, Dict, Any, Literal, Union, Callable, TypeVar
from pydantic import BaseModel, field_validator, ConfigDict, Field
import numpy as np
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


class FrequencyType(str, Enum):
    """Defines how frequently a probe should be run."""

    PRE_TRAINING = "PRE_TRAINING"
    POST_TRAINING = "POST_TRAINING"
    EPOCH = "EPOCH"
    ITERATION = "ITERATION"


class DataSource(str, Enum):
    """Specifies the data source a probe should operate on."""

    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"
    ALL = "ALL"


class ProbeScheduleConfig(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, extra="allow"
    )

    probe_type: str = Field(
        ..., description="Identifier for the probe function in the registry."
    )
    frequency_type: FrequencyType = Field(
        ..., description="When the probe should run (e.g., EPOCH, ITERATION)."
    )
    frequency_value: int = Field(
        1,
        description="Frequency for EPOCH/ITERATION (e.g., run every N epochs/iterations).",
    )
    data_sources: List[DataSource] = Field(
        default_factory=lambda: [DataSource.ALL],
        description="Data sources to run the probe on.",
    )
    probe_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Specific parameters for the probe function."
    )

    @field_validator("frequency_value")
    def check_frequency_value(cls, v, info):
        # Pydantic v2: info is an instance of ValidationInfo
        # Access field values via info.data, which is a dict of the model's fields
        # The .value attribute is not needed for standard enums when use_enum_values=True
        frequency_type_val = info.data.get("frequency_type")
        if frequency_type_val in [
            FrequencyType.EPOCH,
            FrequencyType.ITERATION,
        ]:
            if v < 1:
                raise ValueError(
                    "frequency_value must be at least 1 for EPOCH or ITERATION frequency types."
                )
        return v

    @field_validator("data_sources", mode="before")
    def ensure_data_sources_is_list_and_enum(cls, v):
        if isinstance(v, (str, DataSource)):
            raw_values = [v.value if isinstance(v, DataSource) else v]
        elif isinstance(v, list):
            raw_values = [
                item.value if isinstance(item, DataSource) else item for item in v
            ]
        else:
            # Default if input is not understandable, or raise error
            return [DataSource.TRAIN]

        processed_sources = []
        for val in raw_values:
            try:
                processed_sources.append(DataSource(val))
            except ValueError:
                # Handle cases where the string value might not be a valid DataSource member
                # Depending on strictness, could raise an error or log a warning
                # For now, let's be strict and raise an error or filter out invalid ones
                raise ValueError(f"Invalid DataSource string: {val}")

        if (
            not processed_sources
        ):  # if all were invalid or initial list was empty in a way
            return [DataSource.TRAIN]  # Fallback default
        return processed_sources


class FocalPredictorPreProcessOptions(BaseModel):
    threshold: bool = Field(
        True, description="Whether to apply thresholding on the focal predictor"
    )
    thresholded_value: float = Field(
        0.0, description="Value to use for thresholding the focal predictor"
    )
    interaction_direction: Literal["positive", "negative"] = Field(
        "positive", description="Direction of interaction effect"
    )

    def create_preprocessor(self) -> Callable[[T], T]:
        """
        Creates a function that processes model outputs according to the specified options.
        Supports both numpy arrays and PyTorch tensors.

        Returns:
            A function that takes a numpy array or PyTorch tensor and returns the same type
        """

        def process(focal_predictor: T) -> T:
            # Convert to numpy for processing if it's a tensor
            is_tensor = isinstance(focal_predictor, torch.Tensor)
            if is_tensor:
                focal_predictor_np = focal_predictor.detach().cpu().numpy()
            else:
                focal_predictor_np = focal_predictor

            # Process the array
            if self.threshold:
                if self.interaction_direction == "positive":
                    processed = np.where(
                        focal_predictor_np > self.thresholded_value,
                        focal_predictor_np,
                        0,
                    )
                else:
                    processed = np.where(
                        focal_predictor_np < self.thresholded_value,
                        focal_predictor_np,
                        0,
                    )
            else:
                processed = focal_predictor_np

            # Convert back to tensor if input was a tensor
            if is_tensor:
                return torch.from_numpy(processed).to(focal_predictor.device)
            return processed

        return process


class RegressionEvalProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["regression_eval"] = "regression_eval"
    regress_cmd: Optional[str] = Field(
        None,
        description="Full Stata regression command. If None, it will be generated.",
    )
    evaluation_function: Literal["stata", "statsmodels"] = Field(
        "stata", description="Statistical package to use."
    )
    index_column_name: str = Field(
        "regnn_index",
        description="Column name for the generated ReGNN index in the DataFrame for evaluation.",
    )
    # Replaced Dict with a more specific placeholder if FocalPredictorPreProcessOptions is intended.
    # For now, keeping as Dict[str, Any] to avoid circular dependency or missing import.
    focal_predictor_preprocess_options: Optional[FocalPredictorPreProcessOptions] = (
        Field(
            None,
            description="Options for processing the focal predictor (e.g. thresholding). Corresponds to FocalPredictorPreProcessOptions model.",
        )
    )

    # Removed validator for regress_cmd as it was too basic and might be better handled by the probe itself.
    # If specific validation is needed, it can be reinstated with more robust checks.


class SaveCheckpointProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["save_checkpoint"] = "save_checkpoint"
    save_dir: str = Field(
        "checkpoints", description="Directory to save model checkpoints."
    )
    model_save_name: str = Field(
        "model", description="Base name for the saved model files."
    )
    file_id: Optional[str] = Field(
        None, description="Optional ID to append to filenames."
    )


class SaveIntermediateIndexProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["save_intermediate_index"] = "save_intermediate_index"
    save_dir: str = Field(
        "intermediate_indices",
        description="Directory to save intermediate index files.",
    )
    model_save_name: str = Field(
        "model", description="Base name for the output files (used as prefix)."
    )
    file_id: Optional[str] = Field(
        None, description="Optional ID to append to filenames."
    )
    index_column_name: str = Field(
        description="produced index column name to append to the original dataframe"
    )


class GetObjectiveProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["objective"] = "objective"


class GetL2LengthProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["l2_length"] = "l2_length"


class PValEarlyStoppingProbeScheduleConfig(ProbeScheduleConfig):
    probe_type: Literal["pval_early_stopping"] = "pval_early_stopping"
    frequency_type: FrequencyType = Field(
        default=FrequencyType.EPOCH,  # Use default instead of default_factory for direct enum assignment
        description="Typically runs at EPOCH frequency.",
    )

    criterion: float = Field(
        0.05,
        ge=0,
        le=1,
        description="P-value threshold for stopping (e.g., if p-value < criterion).",
    )
    patience: int = Field(
        0,
        ge=0,
        description="Number of initial epochs to ignore before checking criterion.",
    )
    n_sequential_evals_to_pass: int = Field(
        1,
        ge=1,
        description="Number of consecutive relevant evaluations the criterion must be met.",
    )

    data_sources_to_monitor: List[DataSource] = Field(
        default_factory=lambda: [DataSource.TRAIN, DataSource.TEST],
        description="Data sources (e.g., TRAIN, TEST) whose p-values to monitor from RegressionEvalProbe results.",
    )


# Union type for all specific probe schedule configurations
AllProbeScheduleConfigs = Union[
    RegressionEvalProbeScheduleConfig,
    SaveCheckpointProbeScheduleConfig,
    SaveIntermediateIndexProbeScheduleConfig,
    GetObjectiveProbeScheduleConfig,
    GetL2LengthProbeScheduleConfig,
    PValEarlyStoppingProbeScheduleConfig,
    ProbeScheduleConfig,
]
