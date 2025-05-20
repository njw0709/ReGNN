from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import torch


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    enabled: bool = Field(True, description="Whether to use early stopping")
    criterion: float = Field(0.01, gt=0.0, description="Threshold for early stopping")
    patience: int = Field(
        100, gt=0, description="Number of epochs after which to stop if no improvement"
    )


class LossOptions(BaseModel):
    """Configuration for loss function and regularization"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    weight_decay_regression: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for regression"
    )
    weight_decay_nn: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for neural network"
    )
    use_survey_weights: bool = Field(
        False,
        description="Whether to use survey weights. If true, weighted MSE is used as objective.",
    )


class TrainingHyperParams(BaseModel):
    """Training specific hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Training batch size")
    lr: float = Field(0.001, gt=0.0, description="Learning rate")
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    shuffle: bool = Field(True, description="Whether to shuffle training data")

    stopping_options: Optional[EarlyStoppingConfig] = Field(
        default=None,
        description="Optional early stopping configuration. If None, early stopping will be disabled.",
    )

    loss_options: LossOptions = Field(
        default_factory=LossOptions,
        description="Configuration for loss function and regularization",
    )


class ProbeOptions(BaseModel):
    """Configuration for output and saving behavior"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # File and model saving
    file_id: Optional[str] = Field(None, description="Identifier for saving files")
    save_model: bool = Field(False, description="Whether to save the model")
    model_save_name: str = Field("regnn_", description="model checkpoint save name")
    save_intermediate_index: bool = Field(
        False, description="Whether to save intermediate indices"
    )

    # Output behavior
    return_trajectory: bool = Field(
        False, description="Whether to return training trajectory"
    )
    get_testset_results: bool = Field(
        True, description="Whether to compute results on test set"
    )
    get_l2_lengths: bool = Field(True, description="Whether to compute L2 norms")
