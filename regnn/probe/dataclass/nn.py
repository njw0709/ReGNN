from pydantic import Field, ConfigDict
from typing import Optional, Dict, Literal
from .base import ProbeData


class ObjectiveProbe(ProbeData):
    """Probe for tracking objective/loss values"""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    objective: float = Field(description="The main objective value (e.g., total loss).")
    objective_name: str = Field(
        description="A descriptive name for the objective (e.g., 'total_loss_on_train')."
    )
    objective_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Optional breakdown of the objective (e.g., main loss, regularization loss).",
    )
    probe_type_name: Literal["ObjectiveProbe"] = "ObjectiveProbe"
