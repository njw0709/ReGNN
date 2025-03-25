from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Union, Optional, Sequence
import torch
import numpy as np


class MLPConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    layer_input_sizes: List[int] = Field(
        ..., description="Sizes for each layer including input and output"
    )
    vae: bool = Field(False, description="Whether to use variational autoencoder")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    device: str = Field("cpu", description="Device to run model on")
    output_mu_var: bool = Field(
        False, description="Whether to output mean and variance"
    )
    ensemble: bool = Field(False, description="Whether to use ensemble of models")
    n_ensemble: int = Field(1, ge=1, description="Number of ensemble models")

    @field_validator("layer_input_sizes")
    def validate_layer_sizes(cls, v):
        if len(v) < 2:
            raise ValueError("Must have at least input and output layers")
        return v

    @field_validator("n_ensemble")
    def validate_n_ensemble(cls, v, values):
        if values.get("ensemble", False) and v <= 1:
            raise ValueError("n_ensemble must be greater than 1 when ensemble=True")
        return v


class IndexPredictionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_moderators: Union[int, List[int]] = Field(
        ..., description="Number of moderator variables"
    )
    hidden_layer_sizes: Union[List[int], List[List[int]]] = Field(
        ..., description="Hidden layer sizes"
    )
    svd: bool = Field(False, description="Whether to use SVD")
    svd_matrix: Optional[
        Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
    ] = None
    k_dim: Optional[Union[int, List[int]]] = Field(10, description="SVD dimension")
    batch_norm: bool = Field(True, description="Whether to use batch normalization")
    vae: bool = Field(True, description="Whether to use variational autoencoder")
    device: str = Field("cpu", description="Device to run model on")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    n_ensemble: int = Field(1, ge=1, description="Number of ensemble models")
    output_mu_var: bool = Field(True, description="Whether to output mean and variance")

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

    @field_validator("k_dim")
    def validate_k_dim(cls, v, values):
        if values.data.get("svd", False):
            if v is None:
                raise ValueError("k_dim required when svd=True")
        return v


class ReGNNConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_moderators: Union[int, List[int]] = Field(
        ..., description="Number of moderator variables"
    )
    num_controlled: int = Field(
        ...,
        gt=0,
        description="Number of controlled variables (variables added as linear regression terms)",
    )
    hidden_layer_sizes: Union[List[int], List[List[int]]] = Field(
        ..., description="Hidden layer sizes"
    )
    include_bias_focal_predictor: bool = Field(
        True, description="Whether to include bias for the focal predictor"
    )
    dropout: float = Field(0.5, ge=0.0, le=1.0, description="Dropout rate")
    svd: bool = Field(
        False, description="Whether to use SVD to reduce dim of moderators first"
    )
    svd_matrix: Optional[
        Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
    ] = None
    k_dim: Optional[Union[int, List[int]]] = Field(10, description="SVD dimension")
    device: str = Field("cpu", description="Device to run model on")
    control_moderators: bool = Field(False, description="Whether to control moderators")
    batch_norm: bool = Field(True, description="Whether to use batch normalization")
    vae: bool = Field(False, description="Whether to use variational autoencoder")
    output_mu_var: bool = Field(
        False, description="Whether to output mean and variance"
    )
    interaction_direction: str = Field(
        "positive", description="Direction of interaction effect"
    )
    n_ensemble: int = Field(1, ge=1, description="Number of ensemble models")

    @field_validator("interaction_direction")
    def validate_interaction_direction(cls, v):
        if v not in ["positive", "negative"]:
            raise ValueError("interaction_direction must be 'positive' or 'negative'")
        return v

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

    @field_validator("k_dim")
    def validate_k_dim(cls, v, values):
        if values.data.get("svd", False):
            if v is None:
                raise ValueError("k_dim required when svd=True")
            if isinstance(values.data["num_moderators"], int):
                if v > values.data["num_moderators"]:
                    raise ValueError("k_dim must be <= num_moderators")
            else:
                for i, k in enumerate(v):
                    if k > values.data["num_moderators"][i]:
                        raise ValueError(f"k_dim[{i}] must be <= num_moderators[{i}]")
        return v
