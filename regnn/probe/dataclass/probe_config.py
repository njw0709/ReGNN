from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator, ConfigDict

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
