from typing import Optional, Literal, Union, List, Tuple, Annotated
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
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


class StepLRConfig(BaseModel):
    """Configuration for StepLR scheduler"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["step"] = "step"
    step_size: int = Field(
        30, gt=0, description="Period of learning rate decay (in epochs)"
    )
    gamma: float = Field(
        0.1, gt=0.0, le=1.0, description="Multiplicative factor of learning rate decay"
    )


class ExponentialLRConfig(BaseModel):
    """Configuration for ExponentialLR scheduler"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["exponential"] = "exponential"
    gamma: float = Field(
        0.95, gt=0.0, le=1.0, description="Multiplicative factor of learning rate decay"
    )


class CosineAnnealingLRConfig(BaseModel):
    """Configuration for CosineAnnealingLR scheduler"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["cosine"] = "cosine"
    T_max: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of iterations (defaults to total epochs if None)",
    )
    eta_min: float = Field(
        0.0, ge=0.0, description="Minimum learning rate"
    )


class ReduceLROnPlateauConfig(BaseModel):
    """Configuration for ReduceLROnPlateau scheduler"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["plateau"] = "plateau"
    mode: Literal["min", "max"] = Field(
        "min", description="Whether to minimize or maximize the monitored metric"
    )
    factor: float = Field(
        0.1, gt=0.0, lt=1.0, description="Factor by which the learning rate will be reduced"
    )
    patience: int = Field(
        10, ge=0, description="Number of epochs with no improvement before reducing LR"
    )
    threshold: float = Field(
        1e-4, gt=0.0, description="Threshold for measuring the new optimum"
    )
    threshold_mode: Literal["rel", "abs"] = Field(
        "rel", description="One of rel, abs for relative or absolute threshold"
    )
    cooldown: int = Field(
        0, ge=0, description="Number of epochs to wait before resuming normal operation"
    )
    min_lr: float = Field(
        0.0, ge=0.0, description="Lower bound on the learning rate"
    )


class WarmupCosineConfig(BaseModel):
    """Configuration for Linear Warmup followed by Cosine Annealing"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = Field(
        5, gt=0, description="Number of epochs for linear warmup"
    )
    T_max: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of iterations for cosine (defaults to total epochs - warmup if None)",
    )
    eta_min: float = Field(
        0.0, ge=0.0, description="Minimum learning rate after warmup"
    )


SchedulerConfigUnion = Union[
    StepLRConfig,
    ExponentialLRConfig,
    CosineAnnealingLRConfig,
    ReduceLROnPlateauConfig,
    WarmupCosineConfig,
]


class TemperatureAnnealingConfig(BaseModel):
    """Configuration for temperature/sharpness annealing in SoftTree models"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    schedule_type: Literal["linear", "exponential", "cosine", "step"] = Field(
        "linear", description="Type of temperature annealing schedule"
    )
    initial_temp: float = Field(
        1.0, gt=0.0, description="Starting temperature/sharpness value"
    )
    final_temp: float = Field(
        10.0, gt=0.0, description="Ending temperature/sharpness value"
    )
    # Schedule-specific parameters
    step_size: Optional[int] = Field(
        None,
        gt=0,
        description="For 'step' schedule: period between temperature increases (in epochs)",
    )
    step_gamma: Optional[float] = Field(
        None,
        gt=0.0,
        description="For 'step' schedule: multiplicative factor for temperature increase",
    )
    exp_gamma: Optional[float] = Field(
        None,
        gt=0.0,
        description="For 'exponential' schedule: base for exponential growth",
    )

    @model_validator(mode="after")
    def validate_schedule_params(self):
        """Validate that schedule-specific parameters are provided"""
        if self.schedule_type == "step":
            if self.step_size is None:
                raise ValueError(
                    "step_size must be provided when schedule_type is 'step'"
                )
            if self.step_gamma is None:
                raise ValueError(
                    "step_gamma must be provided when schedule_type is 'step'"
                )
        elif self.schedule_type == "exponential":
            if self.exp_gamma is None:
                raise ValueError(
                    "exp_gamma must be provided when schedule_type is 'exponential'"
                )
        return self


class StepBatchSizeConfig(BaseModel):
    """Configuration for step-based batch size scheduling"""

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["step"] = "step"
    step_size: int = Field(
        ..., gt=0, description="Period of batch size increase (in epochs)"
    )
    gamma: int = Field(
        ..., gt=1, description="Multiplicative factor for batch size increase (must be integer >= 2)"
    )
    max_batch_size: Optional[int] = Field(
        None, gt=0, description="Maximum batch size (optional cap)"
    )


class PiecewiseBatchSizeConfig(BaseModel):
    """Configuration for piecewise batch size scheduling.

    Allows specifying arbitrary batch sizes at specific epoch milestones.
    Useful for warm-start strategies (e.g., start with large batch for stable
    regression coefficients, then switch to smaller batch for better NN generalization).

    Example:
        milestones=[(0, 2048), (50, 256)] means:
        - Epochs 0-49: batch_size = 2048
        - Epochs 50+: batch_size = 256
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)
    type: Literal["piecewise"] = "piecewise"
    milestones: List[Tuple[int, int]] = Field(
        ...,
        description=(
            "List of (epoch, batch_size) tuples defining the batch size schedule. "
            "Must be sorted by epoch. At each epoch, the batch size from the last "
            "milestone at or before that epoch is used."
        ),
    )

    @field_validator("milestones")
    @classmethod
    def validate_milestones(cls, v: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(v) < 1:
            raise ValueError("milestones must contain at least one entry")
        # Sort by epoch
        v = sorted(v, key=lambda x: x[0])
        # Validate epochs are non-negative and batch sizes are positive
        for epoch, batch_size in v:
            if epoch < 0:
                raise ValueError(f"Milestone epoch must be >= 0, got {epoch}")
            if batch_size < 1:
                raise ValueError(f"Milestone batch_size must be >= 1, got {batch_size}")
        # Check for duplicate epochs
        epochs = [e for e, _ in v]
        if len(epochs) != len(set(epochs)):
            raise ValueError("Duplicate epochs in milestones are not allowed")
        return v


BatchSizeSchedulerConfigUnion = Annotated[
    Union[StepBatchSizeConfig, PiecewiseBatchSizeConfig],
    Field(discriminator="type"),
]


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    weight_decay: WeightDecayConfig = Field(
        default_factory=WeightDecayConfig, description="weight decay configurations"
    )
    lr: LearningRateConfig = Field(
        default_factory=LearningRateConfig, description="learning rate configurations"
    )
    scheduler: Optional[SchedulerConfigUnion] = Field(
        None, description="Learning rate scheduler configuration (applies to all parameter groups if per-group schedulers not specified)"
    )
    scheduler_nn: Optional[SchedulerConfigUnion] = Field(
        None,
        description="Learning rate scheduler for NN parameters (overrides scheduler if set)"
    )
    scheduler_regression: Optional[SchedulerConfigUnion] = Field(
        None,
        description="Learning rate scheduler for regression parameters (overrides scheduler if set)"
    )
    temperature_annealing: Optional[TemperatureAnnealingConfig] = Field(
        None, description="Temperature annealing configuration for SoftTree models"
    )

    @model_validator(mode="after")
    def validate_scheduler_config(self):
        """Validate scheduler configuration and warn if mixing global and per-group schedulers"""
        has_global = self.scheduler is not None
        has_per_group = self.scheduler_nn is not None or self.scheduler_regression is not None
        
        if has_global and has_per_group:
            import warnings
            warnings.warn(
                "Both global 'scheduler' and per-group schedulers (scheduler_nn/scheduler_regression) are set. "
                "Per-group schedulers will take precedence over the global scheduler.",
                UserWarning
            )
        return self


class MSELossConfig(LossConfigs):
    name: str = "MSE"


class KLDLossConfig(LossConfigs):
    name: str = "KLDLoss"
    lamba_reg: float = Field(0.01, ge=0, description="kld lambda (mse + lambda*kld)")


class TreeLossConfig(LossConfigs):
    """MSE Loss with SoftTree routing regularization (Frosst & Hinton)"""
    name: str = "TreeLoss"
    lambda_tree: float = Field(
        0.01,
        ge=0,
        description="Lambda weight for tree routing regularization (encourages 50/50 splits)"
    )


class PriorPenaltyLossConfig(LossConfigs):
    """MSE Loss with L2 regularization on treatment effect predictions (prior penalty)"""
    name: str = "PriorPenalty"
    lambda_prior: float = Field(
        0.01,
        ge=0,
        description="Lambda weight for L2 penalty on treatment effects (shrinkage toward zero)"
    )


class TrainingHyperParams(BaseModel):
    """Training specific hyperparameters"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Initial training batch size")
    batch_size_scheduler: Optional[BatchSizeSchedulerConfigUnion] = Field(
        None, description="Batch size scheduler configuration (step-based or piecewise)"
    )
    regression_gradient_accumulation_samples: int = Field(
        1,
        ge=1,
        description=(
            "Number of samples over which to accumulate gradients for regression (MMR) "
            "parameters before applying an update. NN parameters are updated every step "
            "regardless. When the accumulated sample count reaches this target, regression "
            "params are updated with the averaged gradient. This is batch-size-invariant: "
            "if the batch size changes (e.g. via batch_size_scheduler), the same amount of "
            "data is seen before each regression update. Set to 1 (default) to update "
            "every batch."
        ),
    )
    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="optimizer configuration (i.e. weight decay, learning rate)",
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    shuffle: bool = Field(True, description="Whether to shuffle training data")
    kfold: Union[None, int] = Field(
        None, description="Whether to use k-fold division of data"
    )
    k_to_hold: Union[None, int] = Field(None, description="which Kth fold to hold out")
    train_test_split_ratio: float = Field(
        0.8, description="ratio of samples to use for training set"
    )
    val_ratio: float = Field(
        0.0, description="validation dataset ratio. Must be smaller than test"
    )
    train_test_split_seed: int = Field(
        42,
        description="seed to use for shuffling indices to use for splitting train and test set",
    )

    loss_options: LossConfigs = Field(
        default_factory=LossConfigs,
        description="Configuration for loss function and regularization",
    )

    @model_validator(mode="after")
    def check_kfold_and_k(cls, values):
        if values.kfold is not None:
            if values.k_to_hold is None or not isinstance(values.k_to_hold, int):
                raise ValueError(
                    "If kfold is True, 'k' must be provided and must be an integer."
                )
        return values

    @model_validator(mode="after")
    def check_val_ratio(cls, values):
        if values.val_ratio > 0.0:
            if 1 - values.train_test_split_ratio < values.val_ratio:
                raise ValueError(
                    "Test split ratio must be smaller than validation set ratio!"
                )
        return values
