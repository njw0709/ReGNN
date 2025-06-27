from typing import Literal, Optional, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict, computed_field
from .probe_config import AllProbeScheduleConfigs, DataSource


class ProbeData(BaseModel):
    """Base class for all probe data classes"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,  # Allow conversion from objects with attributes
    )

    data_source: Union[DataSource, Literal["NONE"]] = Field(
        ...,
        description="Which data source has been used for computing the probe data. Must be one of: train, test, validate, all",
    )

    status: Literal["success", "failure", "skipped", "error"] = Field(
        "success", description="Execution status of the probe."
    )
    message: Optional[str] = Field(
        None, description="Optional message, e.g., error details if status is failure."
    )

    @computed_field(return_type=str)
    @property
    def probe_type_name(self) -> str:
        return self.__class__.__name__


class ProbeOptions(BaseModel):
    """Configuration for output and saving behavior"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    schedules: List[AllProbeScheduleConfigs] = Field(
        default_factory=list,
        description="List of probe schedules to run. For p-value early stopping, add PValEarlyStoppingProbeScheduleConfig.",
    )

    return_trajectory: bool = Field(
        False, description="Whether to return training trajectory"
    )
