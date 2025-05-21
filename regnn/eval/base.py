from typing import Literal, Optional, Tuple, Callable, Union, TypeVar, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


class FocalPredictorPreProcessOptions(BaseModel):
    threshold: bool = Field(
        True, description="Whether to apply thresholding on the focal predictor"
    )
    thresholded_value: float = Field(
        0.0, description="Value to use for thresholding the focal predictor"
    )
    interaction_direction: Literal["positive", "negative"] = Field(
        "positive", description="Direction of interaction effect"
    )

    def create_preprocessor(self) -> Callable[[T], T]:
        """
        Creates a function that processes model outputs according to the specified options.
        Supports both numpy arrays and PyTorch tensors.

        Returns:
            A function that takes a numpy array or PyTorch tensor and returns the same type
        """

        def process(focal_predictor: T) -> T:
            # Convert to numpy for processing if it's a tensor
            is_tensor = isinstance(focal_predictor, torch.Tensor)
            if is_tensor:
                focal_predictor_np = focal_predictor.detach().cpu().numpy()
            else:
                focal_predictor_np = focal_predictor

            # Process the array
            if self.threshold:
                if self.interaction_direction == "positive":
                    processed = np.where(
                        focal_predictor_np > self.thresholded_value,
                        focal_predictor_np,
                        0,
                    )
                else:
                    processed = np.where(
                        focal_predictor_np < self.thresholded_value,
                        focal_predictor_np,
                        0,
                    )
            else:
                processed = focal_predictor_np

            # Convert back to tensor if input was a tensor
            if is_tensor:
                return torch.from_numpy(processed).to(focal_predictor.device)
            return processed

        return process


class RegressionEvalOptions(BaseModel):
    """Input configuration for evaluation functions"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    regress_cmd: str = Field(..., description="Regression command to run")

    evaluation_function: Literal["stata", "statsmodels"] = Field(
        "statsmodels", description="Which evaluation function to use"
    )
    evaluate: bool = Field(False, description="Whether to evaluate during training")
    eval_epochs: int = Field(10, gt=1, description="Frequency of evaluation in epochs")
    focal_predictor_process_options: FocalPredictorPreProcessOptions = Field(
        FocalPredictorPreProcessOptions(
            threshold=False, thresholded_value=0.0, interaction_direction="positive"
        ),
        description="Sets how to preprocess focal predictor",
    )
    index_column_name: str = Field(
        "summary_index", description="name of the produced summary index"
    )
    post_training_eval: bool = Field(
        True,
        description="whether to run regression on whole dataset after the training is done",
    )

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
