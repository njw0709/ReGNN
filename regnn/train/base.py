from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch
from regnn.eval.base import EvaluationOptions
from regnn.model.base import ReGNNConfig


class TrainingHyperParams(BaseModel):
    """Training specific hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Training batch size")
    lr: float = Field(0.001, gt=0.0, description="Learning rate")
    weight_decay_regression: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for regression"
    )
    weight_decay_nn: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for neural network"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    shuffle: bool = Field(True, description="Whether to shuffle training data")


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    enabled: bool = Field(True, description="Whether to use early stopping")
    criterion: float = Field(0.01, gt=0.0, description="Threshold for early stopping")
    patience: int = Field(
        100, gt=0, description="Number of epochs after which to stop if no improvement"
    )


class OutputOptions(BaseModel):
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


class TrainingConfig(BaseModel):
    """Configuration for ReGNN training"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # Core configurations
    model: ReGNNConfig = Field(..., description="Model configuration")
    training: TrainingHyperParams = Field(
        default_factory=TrainingHyperParams, description="Training hyperparameters"
    )
    evaluation: EvaluationOptions = Field(..., description="Evaluation configuration")
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig, description="Early stopping configuration"
    )
    output: OutputOptions = Field(
        default_factory=OutputOptions, description="Output and saving configuration"
    )

    # Additional settings
    survey_weights: bool = Field(
        False,
        description="Whether to use survey weights. If true, weighted MSE is used as objective.",
    )
