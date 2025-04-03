from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Union, Optional, Sequence
import torch
import numpy as np


class MLPConfig(BaseModel):
    """Base configuration for all neural network architectures"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_layer_sizes: Union[List[int], List[List[int]]] = Field(
        ..., description="Hidden layer sizes"
    )
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    device: str = Field("cpu", description="Device to run model on")
    batch_norm: bool = Field(True, description="Whether to use batch normalization")
    vae: bool = Field(False, description="Whether to use variational autoencoder")
    output_mu_var: bool = Field(
        False, description="Whether to output mean and variance"
    )
    n_ensemble: int = Field(1, ge=1, description="Number of ensemble models")

    @field_validator("hidden_layer_sizes")
    def validate_hidden_layer_sizes(cls, v):
        if isinstance(v, list):
            if isinstance(v[0], list):
                for hidden_layer_sizes in v:
                    if hidden_layer_sizes[-1] != 1:
                        raise ValueError("Last layer of hidden_layer_sizes must be 1")
            else:
                if v[-1] != 1:
                    raise ValueError("Last layer of hidden_layer_sizes must be 1")
        return v


class SVDConfig(BaseModel):
    """Configuration for SVD dimensionality reduction"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(False, description="Whether to use SVD")
    k_dim: Optional[Union[int, List[int]]] = Field(10, description="SVD dimension")
    svd_matrix: Optional[
        Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
    ] = None

    @field_validator("k_dim")
    def validate_k_dim(cls, v, values):
        if values.data.get("enabled", False) and v is None:
            raise ValueError("k_dim required when SVD is enabled")
        return v


class IndexPredictionConfig(MLPConfig):
    """Configuration for index prediction component"""

    num_moderators: Union[int, List[int]] = Field(
        ..., description="Number of moderator variables"
    )
    svd: SVDConfig = Field(default_factory=SVDConfig)

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
    index_prediction: IndexPredictionConfig
    num_controlled: int = Field(..., gt=0, description="Number of controlled variables")

    # Additional settings
    include_bias_focal_predictor: bool = Field(
        True, description="Whether to include bias for focal predictor"
    )
    control_moderators: bool = Field(False, description="Whether to control moderators")
    interaction_direction: str = Field(
        "positive", description="Direction of interaction effect"
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
        hidden_layer_sizes: Union[List[int], List[List[int]]],
        **kwargs,
    ) -> "ReGNNConfig":
        """Factory method for creating ReGNN config with default values"""
        index_prediction = IndexPredictionConfig(
            num_moderators=num_moderators, hidden_layer_sizes=hidden_layer_sizes
        )
        return cls(
            index_prediction=index_prediction, num_controlled=num_controlled, **kwargs
        )
