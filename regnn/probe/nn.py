from pydantic import Field, ConfigDict
from .base import ProbeData


class ObjectiveProbe(ProbeData):
    """Probe for tracking objective/loss values"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    loss: float = Field(-1, description="loss")
