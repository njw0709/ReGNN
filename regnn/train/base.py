from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
import torch


class RegularizationConfig(BaseModel):
    """Configuration for regularization"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    name: str = Field(..., description="name of the regularization function used")
    regularization_alpha: float = Field(
        0.1, description="alpha weight for regularization term"
    )


class ElasticNetRegConfig(RegularizationConfig):
    name: str = "elasticnet"
    elastic_net_alpha: float = Field(
        0.1,
        description="alpha balancing l2 and l1 regularization. (1-alpha)*l2 + alpha*l1",
    )


class LossConfigs(BaseModel):
    """Configuration for loss function"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    name: str = Field("MSE", description="name of the loss function used")
    reduction: Literal["mean", "sum", "none"] = Field(
        "mean", description="how loss is reduced per batch"
    )
    regularization: Optional[RegularizationConfig] = Field(
        None, description="regularization configuration"
    )


class WeightDecayConfig(BaseModel):
    """Weight decay specific hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # weight decay option
    weight_decay_regression: Optional[float] = Field(
        0.0, ge=0.0, description="L2 regularization weight for regression"
    )
    weight_decay_nn: Optional[float] = Field(
        0.0, ge=0.0, description="L2 regularization weight for neural network"
    )


class LearningRateConfig(BaseModel):
    """learning rate hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # learning rate options
    lr_regression: Optional[float] = Field(
        0.001, ge=0.0, description="learning rate for regression coefficients"
    )
    lr_nn: Optional[float] = Field(
        0.001, ge=0.0, description="learning rate for neural net"
    )


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    weight_decay: WeightDecayConfig = Field(
        default_factory=WeightDecayConfig, description="weight decay configurations"
    )
    lr: LearningRateConfig = Field(
        default_factory=LearningRateConfig, description="learning rate configurations"
    )


class MSELossConfig(LossConfigs):
    name: str = "MSE"


class KLDLossConfig(LossConfigs):
    name: str = "KLDLoss"
    lamba_reg: float = Field(0.01, ge=0, description="kld lambda (mse + lambda*kld)")


class TrainingHyperParams(BaseModel):
    """Training specific hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Training batch size")
    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="optimizer configuration (i.e. weight decay, learning rate)",
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    shuffle: bool = Field(True, description="Whether to shuffle training data")
    train_test_split_ratio: float = Field(
        0.8, description="ratio of samples to use for training set"
    )
    train_test_split_seed: int = Field(
        42,
        description="seed to use for shuffling indices to use for splitting train and test set",
    )

    loss_options: LossConfigs = Field(
        default_factory=LossConfigs,
        description="Configuration for loss function and regularization",
    )
