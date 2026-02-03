from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
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

    @property
    def hidden_layer_sizes(self) -> Union[List[int], List[List[int]]]:
        """Computed property that returns layer_input_sizes with 1 appended to each list"""
        if isinstance(self.layer_input_sizes[0], list):
            return [sizes + [1] for sizes in self.layer_input_sizes]
        return self.layer_input_sizes + [1]


class TreeConfig(BaseModel):
    """Configuration for SoftTree architecture"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dim: int = Field(..., description="Input feature dimension")
    output_dim: int = Field(..., description="Output dimension")
    depth: int = Field(5, ge=1, description="Tree depth (number of levels)")
    sharpness: float = Field(
        1.0, gt=0.0, description="Sharpness of sigmoid routing (higher = sharper)"
    )
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    batch_norm: bool = Field(
        False,
        description="Whether to apply batch normalization to output (without affine parameters)",
    )
    device: str = Field("cpu", description="Device to run model on")


class IndexPredictionConfig(MLPConfig):
    """Configuration for index prediction component"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    num_moderators: Union[int, List[int]] = Field(
        ..., description="Number of moderator variables"
    )
    n_ensemble: int = Field(1, ge=1, description="Number of models to ensemble")
    use_resmlp: bool = Field(
        False, description="Whether to use ResidualMLP instead of MLP"
    )
    use_soft_tree: bool = Field(
        False, description="Whether to use SoftTree instead of MLP/ResMLP"
    )
    tree_depth: Optional[int] = Field(
        None, ge=1, description="Tree depth when use_soft_tree=True"
    )
    tree_sharpness: float = Field(
        1.0,
        gt=0.0,
        description="Sharpness of sigmoid routing in SoftTree (higher = sharper)",
    )

    @model_validator(mode="after")
    def validate_backbone_selection(self):
        # Ensure mutual exclusivity between use_resmlp and use_soft_tree
        if self.use_resmlp and self.use_soft_tree:
            raise ValueError(
                "Cannot use both use_resmlp and use_soft_tree simultaneously"
            )

        # Ensure tree_depth is provided when using SoftTree
        if self.use_soft_tree and self.tree_depth is None:
            raise ValueError("tree_depth must be provided when use_soft_tree=True")

        # VAE is not supported with SoftTree
        if self.use_soft_tree and self.vae:
            raise ValueError(
                "VAE is not supported with SoftTree. Set vae=False when use_soft_tree=True"
            )

        return self


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
            "n_ensemble",
            "use_resmlp",
            "use_soft_tree",
            "tree_depth",
            "tree_sharpness",
        }

        # Split kwargs between nn_config and main config
        nn_kwargs = {k: v for k, v in kwargs.items() if k in nn_params}
        main_kwargs = {k: v for k, v in kwargs.items() if k not in nn_params}

        nn_config = IndexPredictionConfig(
            num_moderators=num_moderators,
            layer_input_sizes=layer_input_sizes,
            **nn_kwargs,
        )
        return cls(nn_config=nn_config, num_controlled=num_controlled, **main_kwargs)
