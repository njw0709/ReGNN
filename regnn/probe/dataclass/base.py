from typing import Literal
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

    @computed_field(return_type=str)
    @property
    def __type__(self) -> str:
        return self.__class__.__name__
