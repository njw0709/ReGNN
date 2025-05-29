from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Union, Optional
import torch


class MLPConfig(BaseModel):
    """Base configuration for all neural network architectures"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    layer_input_sizes: Union[List[int], List[List[int]]] = Field(
        ..., description="input sizes of the layers"
    )
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    device: str = Field("cpu", description="Device to run model on")
    batch_norm: bool = Field(True, description="Whether to use batch normalization")
    vae: bool = Field(False, description="Whether to use variational autoencoder")
    output_mu_var: bool = Field(
        False, description="Whether to output mean and variance"
    )
    ensemble: bool = Field(
        False, description="Whether the model is a part of an ensemble"
    )

    @property
    def hidden_layer_sizes(self) -> Union[List[int], List[List[int]]]:
        """Computed property that returns layer_input_sizes with 1 appended to each list"""
        if isinstance(self.layer_input_sizes[0], list):
            return [sizes + [1] for sizes in self.layer_input_sizes]
        return self.layer_input_sizes + [1]


class SVDConfig(BaseModel):
    """Configuration for SVD dimensionality reduction"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(False, description="Whether to use SVD")
    k_dim: Optional[Union[int, List[int]]] = Field(10, description="SVD dimension")
    svd_matrix: Optional[torch.Tensor] = Field(
        None, description="Precomputed or computed SVD matrix"
    )

    @field_validator("k_dim")
    def validate_k_dim(cls, v, values):
        if values.data.get("enabled", False) and v is None:
            raise ValueError("k_dim required when SVD is enabled")
        return v


class IndexPredictionConfig(MLPConfig):
    """Configuration for index prediction component"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    num_moderators: Union[int, List[int]] = Field(
        ..., description="Number of moderator variables"
    )
    n_ensemble: int = Field(1, ge=1, description="Number of MLP models to ensemble")
    svd: SVDConfig = Field(default_factory=SVDConfig)

    @field_validator("n_ensemble")
    def validate_ensemble(cls, v, values):
        if v > 1 and not values.data.get("ensemble", False):
            raise ValueError("ensemble must be True when n_ensemble > 1")
        return v

    @field_validator("svd")
    def validate_svd_k_dim(cls, v, values):
        if v.enabled and isinstance(values.data.get("num_moderators"), int):
            if v.k_dim > values.data["num_moderators"]:
                raise ValueError("k_dim must be <= num_moderators")
        return v


class ReGNNConfig(BaseModel):
    """Main configuration for ReGNN model"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core components
    nn_config: IndexPredictionConfig
    num_controlled: int = Field(..., ge=0, description="Number of controlled variables")

    # Additional settings
    include_bias_focal_predictor: bool = Field(
        True, description="Whether to include bias for focal predictor"
    )
    control_moderators: bool = Field(False, description="Whether to control moderators")
    interaction_direction: str = Field(
        "positive", description="Direction of interaction effect"
    )
    use_closed_form_linear_weights: bool = Field(
        False,
        description="Whether to update linear weights using closed form solution or gradient descent",
    )

    @field_validator("interaction_direction")
    def validate_interaction_direction(cls, v):
        if v not in ["positive", "negative"]:
            raise ValueError("interaction_direction must be 'positive' or 'negative'")
        return v

    @classmethod
    def create(
        cls,
        num_moderators: Union[int, List[int]],
        num_controlled: int,
        layer_input_sizes: Union[List[int], List[List[int]]],
        **kwargs,
    ) -> "ReGNNConfig":
        """Factory method for creating ReGNN config with default values"""
        # Parameters that should go to IndexPredictionConfig
        nn_params = {
            "dropout",
            "device",
            "batch_norm",
            "vae",
            "output_mu_var",
            "ensemble",
            "svd",
        }

        # Split kwargs between nn_config and main config
        nn_kwargs = {k: v for k, v in kwargs.items() if k in nn_params}
        # svd_kwargs = {k: v for k, v in kwargs.items() if k in svd_params}
        main_kwargs = {k: v for k, v in kwargs.items() if k not in nn_params}

        nn_config = IndexPredictionConfig(
            num_moderators=num_moderators,
            layer_input_sizes=layer_input_sizes,
            **nn_kwargs,
        )
        return cls(nn_config=nn_config, num_controlled=num_controlled, **main_kwargs)
