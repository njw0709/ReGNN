from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, computed_field


class ProbeData(BaseModel):
    """Base class for all probe data classes"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,  # Allow conversion from objects with attributes
    )

    data_source: Literal["train", "test", "validate", "all"] = Field(
        ...,
        description="Which data source has been used for computing the probe data. Must be one of: train, test, validate, all",
    )

    status: Literal["success", "failure", "skipped"] = Field(
        "success", description="Execution status of the probe."
    )
    message: Optional[str] = Field(
        None, description="Optional message, e.g., error details if status is failure."
    )
    value: Optional[Any] = Field(
        None, description="Generic field for a simple, primary result of the probe."
    )

    @computed_field(return_type=str)
    @property
    def probe_type_name(self) -> str:
        return self.__class__.__name__
