from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch


class TrainingConfig(BaseModel):
    """Configuration for ReGNN training"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    hidden_layer_sizes: List[int] = Field(
        ..., description="Sizes for each layer in the neural network"
    )
    vae: bool = Field(False, description="Whether to use variational autoencoder")
    svd: bool = Field(False, description="Whether to use SVD dimensionality reduction")
    k_dims: int = Field(10, gt=0, description="Number of dimensions for SVD reduction")
    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Training batch size")
    lr: float = Field(0.001, gt=0.0, description="Learning rate")
    weight_decay_regression: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for regression"
    )
    weight_decay_nn: float = Field(
        0.0, ge=0.0, description="L2 regularization weight for neural network"
    )
    regress_cmd: str = Field("", description="Command for regression")
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    shuffle: bool = Field(True, description="Whether to shuffle training data")
    evaluate: bool = Field(False, description="Whether to evaluate during training")
    eval_epoch: int = Field(10, gt=0, description="Frequency of evaluation in epochs")
    get_testset_results: bool = Field(
        True, description="Whether to compute results on test set"
    )
    file_id: Optional[str] = Field(0, description="Identifier for saving files")
    save_model: bool = Field(False, description="Whether to save the model")
    use_stata: bool = Field(False, description="Whether to use Stata for analysis")
    return_trajectory: bool = Field(
        False, description="Whether to return training trajectory"
    )
    vae_loss: bool = Field(False, description="Whether to use VAE loss")
    vae_lambda: float = Field(0.1, gt=0.0, description="Weight for VAE loss term")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    n_models: int = Field(1, gt=0, description="Number of models in ensemble")
    elasticnet: bool = Field(
        False, description="Whether to use elastic net regularization"
    )
    lasso: bool = Field(False, description="Whether to use LASSO regularization")
    lambda_reg: float = Field(0.1, ge=0.0, description="Regularization strength")
    survey_weights: bool = Field(True, description="Whether to use survey weights")
    include_bias_focal_predictor: bool = Field(
        True, description="Whether to include bias for focal predictor"
    )
    interaction_direction: str = Field(
        "positive", description="Direction of interaction effect"
    )
    get_l2_lengths: bool = Field(True, description="Whether to compute L2 norms")
    early_stop: bool = Field(True, description="Whether to use early stopping")
    early_stop_criterion: float = Field(
        0.01, gt=0.0, description="Threshold for early stopping"
    )
    stop_after: int = Field(
        100, gt=0, description="Number of epochs after which to stop if no improvement"
    )
    save_intermediate_index: bool = Field(
        False, description="Whether to save intermediate indices"
    )
    model_save_name: str = Field("regnn_", description="model checkpoint save name")

    @field_validator("hidden_layer_sizes")
    def validate_hidden_layer_sizes(cls, v):
        if not v:
            raise ValueError("hidden_layer_sizes cannot be empty")

        # Check if it's a list of lists
        if isinstance(v[0], list):
            for layer_sizes in v:
                if not layer_sizes:
                    raise ValueError("Layer sizes cannot be empty")
                if layer_sizes[-1] != 1:
                    raise ValueError("Last layer size must be 1 for all models")
        else:
            # Single model case
            if v[-1] != 1:
                raise ValueError("Last layer size must be 1")
        return v

    @field_validator("interaction_direction")
    def validate_interaction_direction(cls, v):
        if v not in ["positive", "negative"]:
            raise ValueError("interaction_direction must be 'positive' or 'negative'")
        return v


class TrajectoryData(BaseModel):
    """Data structure for tracking training trajectory"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    train_loss: float = Field(-1, description="Training loss")
    test_loss: float = Field(-1, description="Test loss")
    regression_summary: Optional[dict] = Field(
        None, description="Summary of regression results on training data"
    )
    regression_summary_test: Optional[dict] = Field(
        None, description="Summary of regression results on test data"
    )
    l2: Optional[List[dict]] = Field(None, description="L2 norms during training")
