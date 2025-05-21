from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field
from .preprocess_fns import (
    binary_to_one_hot,
    multi_cat_to_one_hot,
    convert_categorical_to_ordinal,
    standardize_cols,
)

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


class DataFrameReadInConfig(BaseModel):
    """Configurations / variables needed to read in the dataframe."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    data_path: str
    read_cols: Sequence[str]
    rename_dict: dict
    binary_cols: List[str]
    categorical_cols: List[str]
    ordinal_cols: List[str]
    continuous_cols: List[str]
    survey_weight_col: Optional[str]

    @field_validator(
        "binary_cols",
        "categorical_cols",
        "ordinal_cols",
        "continuous_cols",
        "survey_weight_col",
    )
    @classmethod
    def validate_columns_in_read_cols(
        cls, v: Union[List[str], str], info
    ) -> Union[List[str], str]:
        """Validate that all specified columns are present in read_cols."""
        read_cols = info.data.get("read_cols", [])
        if isinstance(v, str):  # For survey_weight_col
            if v and v not in read_cols:
                raise ValueError(f"Column '{v}' must be present in read_cols")
        else:  # For list columns
            missing_cols = [col for col in v if col not in read_cols]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} must be present in read_cols")
        return v

    @computed_field
    @property
    def df(self) -> pd.DataFrame:
        """Read and return the dataframe."""
        df = pd.read_stata(self.data_path, columns=self.read_cols)
        if self.rename_dict:
            df = df.rename(columns=self.rename_dict)
        return df

    @computed_field
    @property
    def df_dtypes(self) -> Dict[str, List[str]]:
        """Get the dtypes dictionary for the dataframe."""
        df_dtype_list = [("category", c) for c in self.categorical_cols]
        df_dtype_list += [("ordinals", c) for c in self.ordinal_cols]
        df_dtype_list += [("binary", c) for c in self.binary_cols]
        df_dtype_list += [("continuous", c) for c in self.continuous_cols]
        return dict(df_dtype_list)

    @computed_field
    @property
    def preprocess_steps(self) -> List[PreprocessStep]:
        """Get the preprocessing steps for the dataframe."""
        steps = [
            PreprocessStep(columns=self.binary_cols, function=binary_to_one_hot),
            PreprocessStep(
                columns=self.categorical_cols, function=multi_cat_to_one_hot
            ),
            PreprocessStep(
                columns=self.ordinal_cols, function=convert_categorical_to_ordinal
            ),
            PreprocessStep(
                columns=self.continuous_cols + self.ordinal_cols,
                function=standardize_cols,
            ),
        ]

        # Add survey weight standardization if provided
        if self.survey_weight_col:
            steps.append(
                PreprocessStep(
                    columns=[self.survey_weight_col],
                    function=standardize_cols,
                )
            )

        return steps


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
