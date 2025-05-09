from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

numeric = Union[int, float, complex, np.number]


class PreprocessStep(BaseModel):
    """Represents a preprocessing step with columns and function"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    columns: List[str]
    function: Callable
    reverse_function: Optional[Callable] = None
    reverse_transform_info: Dict[str, Any] = {}

    def __init__(self, **data):
        super().__init__(**data)
        # If reverse_function is not provided, try to get it from the function
        if self.reverse_function is None and hasattr(
            self.function, "_reverse_transform"
        ):
            self.reverse_function = self.function._reverse_transform


class ReGNNDatasetConfig(BaseModel):
    """Configuration for ReGNN datasets"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    focal_predictor: str
    controlled_predictors: List[str]
    moderators: Union[List[str], List[List[str]]]
    outcome: str
    survey_weights: Optional[str] = None
    df_dtypes: Dict[str, List[str]]
    rename_dict: Dict[str, str]
    preprocess_steps: List[PreprocessStep]

    @field_validator("moderators")
    @classmethod
    def validate_moderators(cls, v):
        """Validate that there are at least 2 moderators"""
        if isinstance(v[0], str):
            if len(v) < 2:
                raise ValueError("Must have at least 2 moderators")
        else:
            total_moderators = sum(len(sublist) for sublist in v)
            if total_moderators < 2:
                raise ValueError(
                    "Must have at least 2 moderators in total across all lists"
                )
        return v

    @field_validator("focal_predictor", "outcome")
    @classmethod
    def validate_non_empty_strings(cls, v):
        """Validate that strings are not empty"""
        if not v or v.isspace():
            raise ValueError("Cannot be empty or whitespace")
        return v

    @field_validator("controlled_predictors")
    @classmethod
    def validate_controlled_predictors(cls, v):
        """Validate that controlled predictors list is not empty and has no empty strings"""
        if any(not pred or pred.isspace() for pred in v):
            raise ValueError(
                "Controlled predictors cannot contain empty strings or whitespace"
            )
        return v


class BaseDataset:
    """Base class for dataset operations"""

    def __init__(self, df: pd.DataFrame, config: ReGNNDatasetConfig):
        self.df = df
        self.columns = df.columns.tolist()
        self.config = config
        self.mean_std_dict: Dict[str, Tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def dropna(self, inplace: bool = True):
        if inplace:
            self.df = self.df.dropna()
        else:
            return self.df.dropna()

    def get_column_index(self, colname: str) -> int:
        return self.columns.index(colname)
