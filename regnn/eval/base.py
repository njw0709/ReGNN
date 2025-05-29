from typing import Literal, Callable, TypeVar
import numpy as np
from pydantic import BaseModel, Field
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
