from pydantic import (
    Field,
    ConfigDict,
    field_validator,
    BaseModel,
    model_serializer,
    field_serializer,
)
from pydantic.functional_serializers import PlainSerializer
from typing_extensions import Annotated
from numpydantic import NDArray, Shape
from .base import ProbeData
from typing import Dict, Optional, List, Union, Literal, Any
import math
import numpy as np
import json


def serialize_model_class(model_class: Optional[type]) -> Optional[str]:
    """Serialize a model class to its fully qualified name"""
    if model_class is None:
        return None
    return f"{model_class.__module__}.{model_class.__name__}"


class DebiasConfig(BaseModel):
    """Configuration for debiasing the focal predictor"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(True, description="Whether to apply debiasing")
    model_class: Annotated[
        Optional[type],
        PlainSerializer(serialize_model_class, return_type=str, when_used="json"),
    ] = Field(
        None,
        description="Model class for debiasing (e.g., Ridge, RandomForestRegressor). If None, uses RandomForest",
    )
    k: int = Field(5, ge=2, description="Number of folds for cross-validation")
    is_classifier: bool = Field(
        False,
        description="True for binary focal predictor (propensity scores), False for continuous",
    )
    sample_weight_col: Optional[str] = Field(
        None, description="Column name containing sample weights for model fitting"
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters to pass to the model"
    )


class ModeratedRegressionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    focal_predictor: str
    outcome_col: str
    controlled_cols: List[str]
    moderators: Union[List[str], List[List[str]]]
    control_moderators: bool = Field(True)
    index_column_name: Union[str, List[str]]
    debias_treatment: bool = Field(
        False, description="Whether to debias the focal predictor"
    )
    debias_config: Optional[DebiasConfig] = Field(
        None,
        description="Advanced debiasing configuration. If None and debias_treatment=True, uses defaults",
    )


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
