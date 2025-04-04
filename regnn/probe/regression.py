from pydantic import Field, ConfigDict, field_validator
from .base import ProbeData


class RegressionProbe(ProbeData):
    """Output from regression evaluation functions"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    interaction_pval: float = Field(..., description="P-value of the interaction term")
    rsquared: float = Field(..., description="R-squared value")
    adjusted_rsquared: float = Field(..., description="Adjusted R-squared value")
    rmse: float = Field(..., description="Root mean squared error")

    @field_validator("interaction_pval")
    def validate_pval(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("p-value must be between 0 and 1")
        return v

    @field_validator("rsquared", "adjusted_rsquared")
    def validate_rsquared(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("R-squared values must be between 0 and 1")
        return v

    @field_validator("rmse")
    def validate_positive_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class VarianceInflationFactorProbe(ProbeData):
    """Probe for tracking variance inflation factors"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    vif_main: float = Field(
        ..., description="Variance inflation factor for main effect"
    )
    vif_interaction: float = Field(
        ..., description="Variance inflation factor for interaction term"
    )
