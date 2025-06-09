from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import (
    BaseModel,
    field_validator,
    ConfigDict,
    computed_field,
    model_serializer,
    model_validator,
)
from .preprocess_fns import (
    binary_to_one_hot,
    multi_cat_to_one_hot,
    convert_categorical_to_ordinal,
    standardize_cols,
    map_to_zero_one,
)
from collections import defaultdict
from pathlib import Path

numeric = Union[int, float, complex, np.number]

function_registry = {
    "binary_to_one_hot": binary_to_one_hot,
    "multi_cat_to_one_hot": multi_cat_to_one_hot,
    "standardize_cols": standardize_cols,
    "convert_categorical_to_ordinal": convert_categorical_to_ordinal,
    "map_to_zero_one": map_to_zero_one,
}


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

    @model_serializer
    def serialize(self) -> dict:
        return {
            "columns": self.columns,
            "function": self.function.__name__,
            "reverse_transform_info": self.reverse_transform_info,
        }

    @model_validator(mode="before")
    @classmethod
    def deserialize(cls, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Invalid data format for PreprocessStep")

        # If already a function, skip registry lookup
        fn = data.get("function")
        if callable(fn):
            return data

        # Else resolve from string
        function_name = data.get("function")
        if not function_name:
            raise ValueError("Missing 'function' name in PreprocessStep")

        if function_name not in function_registry:
            raise ValueError(f"Unknown function '{function_name}'")

        data = data.copy()
        data["function"] = function_registry[function_name]
        return data


class DataFrameReadInConfig(BaseModel):
    """Configurations / variables needed to read in the dataframe."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    data_path: str
    read_cols: Sequence[str]
    rename_dict: dict
    binary_cols: List[str]
    categorical_cols: List[str]
    ordinal_cols: List[str]
    continuous_cols: List[str]
    survey_weight_col: Optional[str]

    # Mapping of file extensions to their corresponding pandas read functions
    _READ_FUNCTIONS = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".dta": pd.read_stata,
        ".parquet": pd.read_parquet,
        ".feather": pd.read_feather,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
    }

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
        if not v:  # Handle None or empty list/string
            return v
        read_cols = info.data.get("read_cols")
        if not read_cols:  # Handle case where read_cols is None
            raise ValueError("read_cols must be provided")
        if isinstance(v, str):  # For survey_weight_col
            if v not in read_cols:
                raise ValueError(f"Column '{v}' must be present in read_cols")
        else:  # For list columns
            missing_cols = [col for col in v if col not in read_cols]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} must be present in read_cols")
        return v

    def read_df(self) -> pd.DataFrame:
        """Read and return the dataframe based on file extension.

        Supports multiple file formats:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - Stata (.dta)
        - Parquet (.parquet)
        - Feather (.feather)
        - Pickle (.pkl, .pickle)
        """
        file_path = Path(self.data_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        file_ext = file_path.suffix.lower()
        if file_ext not in self._READ_FUNCTIONS:
            raise ValueError(
                f"Unsupported file extension: {file_ext}. "
                f"Supported extensions are: {', '.join(self._READ_FUNCTIONS.keys())}"
            )

        read_func = self._READ_FUNCTIONS[file_ext]

        # Special handling for read_cols parameter based on file type
        kwargs = {}
        if self.read_cols:
            if file_ext in [
                ".csv",
                ".xlsx",
                ".xls",
                ".parquet",
            ]:
                kwargs["usecols"] = self.read_cols
            elif file_ext in [".dta", ".feather"]:
                kwargs["columns"] = self.read_cols
            if file_ext == ".dta":
                kwargs["convert_categoricals"] = False
            # For pickle files, we'll read all columns and filter after

        df = read_func(self.data_path, **kwargs)

        # For pickle files, filter columns after reading
        if file_ext in [".pkl", ".pickle"] and self.read_cols:
            df = df[self.read_cols]

        return df

    @computed_field
    @property
    def df_dtypes(self) -> Dict[str, List[str]]:
        """Get the dtypes dictionary for the dataframe."""
        dtypes = defaultdict(list)
        for col in self.categorical_cols:
            dtypes["category"].append(col)
        for col in self.ordinal_cols:
            dtypes["ordinals"].append(col)
        for col in self.binary_cols:
            dtypes["binary"].append(col)
        for col in self.continuous_cols:
            dtypes["continuous"].append(col)
        return dict(dtypes)

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
                    function=map_to_zero_one,
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
