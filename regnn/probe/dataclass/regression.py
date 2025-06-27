from pydantic import Field, ConfigDict, field_validator, BaseModel
from numpydantic import NDArray, Shape
from .base import ProbeData
from typing import Dict, Optional, List, Union, Literal
import math
import numpy as np


class ModeratedRegressionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    focal_predictor: str
    outcome_col: str
    controlled_cols: List[str]
    moderators: Union[List[str], List[List[str]]]
    control_moderators: bool = Field(True)
    index_column_name: Union[str, List[str]]


class OLSResultsProbe(ProbeData):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    rsquared: Optional[float] = Field(None, description="R-squared value")
    adjusted_rsquared: Optional[float] = Field(
        None, description="Adjusted R-squared value"
    )
    rmse: Optional[float] = Field(None, description="Root mean squared error")

    @field_validator("rsquared", "adjusted_rsquared")
    def validate_rsquared(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("R-squared values must be between 0 and 1")
        return v

    @field_validator("rmse")
    def validate_positive_float(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Value must be positive")
        return v


class OLSModeratedResultsProbe(OLSResultsProbe):
    """Output from regression evaluation functions"""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)
    interaction_pval: float = Field(-1.0, description="P-value of the interaction term")

    # New fields for more comprehensive results
    coefficients: Optional[Union[NDArray[Shape["*"], np.float32], List]] = Field(
        None, description="All estimated coefficients from the regression model."
    )
    standard_errors: Optional[Union[NDArray[Shape["*"], np.float32], List]] = Field(
        None, description="Standard errors for the coefficients."
    )
    p_values: Optional[Union[NDArray[Shape["*"], np.float32], List]] = Field(
        None, description="P-values for the coefficients."
    )
    n_observations: Optional[int] = Field(
        None, ge=0, description="Number of observations used in the regression."
    )
    raw_summary: Optional[str] = Field(
        None,
        description="Raw summary string as output by the statistical package (e.g., statsmodels summary table).",
    )

    @field_validator("interaction_pval")
    def validate_pval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 <= v <= 1) and not math.isnan(v):
            raise ValueError("p-value must be between 0 and 1")
        return v


class VarianceInflationFactorProbe(ProbeData):
    """Probe for tracking variance inflation factors"""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    vif_main: float = Field(
        ..., description="Variance inflation factor for main effect"
    )
    vif_interaction: float = Field(
        ..., description="Variance inflation factor for interaction term"
    )


class L2NormProbe(ProbeData):
    """Probe for tracking L2 norms of model parameters"""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    main_norm: float = Field(-1, description="L2 norm of main parameters")
    index_norm: float = Field(-1, description="L2 norm of index parameters")

    @classmethod
    def from_dict(
        cls, data: Dict[str, float], data_source: str = "train"
    ) -> "L2NormProbe":
        """Create an L2NormProbe from dictionary returned by get_l2_length"""
        return cls(
            data_source=data_source,
            main_norm=data.get("main", -1),
            index_norm=data.get("index", -1),
        )

    @classmethod
    def compute(cls, model, data_source: str = "train") -> "L2NormProbe":
        """Compute L2 norms for model parameters and return a populated L2NormProbe instance.

        Args:
            model: The model to compute L2 norms for
            data_source: Source of data ('train', 'test', 'validate')

        Returns:
            L2NormProbe instance with computed norms
        """
        from regnn.probe import get_l2_length

        l2_data = get_l2_length(model)
        return cls.from_dict(l2_data, data_source=data_source)
