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


class SchedulerProbe(ProbeData):
    """Probe for monitoring learning rate scheduler and temperature annealing"""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    lr_nn: float = Field(
        description="Current learning rate for neural network parameters."
    )
    lr_regression: float = Field(
        description="Current learning rate for regression parameters."
    )
    temperature: Optional[float] = Field(
        None, description="Current temperature/sharpness for SoftTree (if applicable)."
    )
    scheduler_type: Optional[str] = Field(
        None, description="Type of LR scheduler in use (if any)."
    )
