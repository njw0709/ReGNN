from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch


class EvaluationOptions(BaseModel):
    """Input configuration for evaluation functions"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    regress_cmd: str = Field(..., description="Regression command to run")
    save_dir: str = Field(
        "data/temp/data", description="Directory to save intermediate files"
    )
    data_id: Optional[str] = Field(None, description="Identifier for saving files")
    save_intermediate: bool = Field(
        False, description="Whether to save intermediate files"
    )
    threshold: bool = Field(True, description="Whether to apply thresholding")
    thresholded_value: float = Field(0.0, description="Value to use for thresholding")
    interaction_direction: Literal["positive", "negative"] = Field(
        "positive", description="Direction of interaction effect"
    )
    evaluation_function: Literal["stata", "statsmodels"] = Field(
        "statsmodels", description="Which evaluation function to use"
    )
    evaluate: bool = Field(False, description="Whether to evaluate during training")
    eval_epoch: int = Field(10, gt=0, description="Frequency of evaluation in epochs")

    @field_validator("regress_cmd")
    def validate_regress_cmd(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("regress_cmd cannot be empty")
        parts = v.split()
        if len(parts) < 3:
            raise ValueError(
                "regress_cmd must have at least 3 parts: command, dependent var, and independent var"
            )
        return v

    @field_validator("index_predictions")
    def validate_index_predictions(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError("index_predictions must be 1-dimensional")
        return v


# class ReGNNEvalInput(EvaluationOptions):
#     """Input configuration for ReGNN evaluation"""

#     model: ReGNN = Field(..., description="ReGNN model to evaluate")
#     test_dataset: ReGNNDataset = Field(..., description="Test dataset for evaluation")
#     device: str = Field(
#         "cuda" if torch.cuda.is_available() else "cpu",
#         description="Device to run model on",
#     )
#     quietly: bool = Field(True, description="Whether to suppress Stata output")

#     @field_validator("device")
#     def validate_device(cls, v: str) -> str:
#         if v not in ["cuda", "cpu"]:
#             raise ValueError("device must be either 'cuda' or 'cpu'")
#         if v == "cuda" and not torch.cuda.is_available():
#             return "cpu"
#         return v

#     @field_validator("test_dataset")
#     def validate_test_dataset(cls, v: ReGNNDataset) -> ReGNNDataset:
#         if len(v) == 0:
#             raise ValueError("test_dataset cannot be empty")
#         return v
