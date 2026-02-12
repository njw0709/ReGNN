import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Union, Sequence, List
import numpy as np
from torch.nn.utils import spectral_norm
from .base import MLPConfig, IndexPredictionConfig, ReGNNConfig, TreeConfig


class MLP(nn.Module):
    """Simple feedforward neural network with GELU activations and optional dropout."""

    @classmethod
    def from_config(cls, config: MLPConfig):
        return cls(
            layer_input_sizes=config.layer_input_sizes,
            dropout=config.dropout,
            device=config.device,
        )

    def __init__(
        self,
        layer_input_sizes: List[int],
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super(MLP, self).__init__()
        self.layer_input_sizes = layer_input_sizes
        self.num_layers = len(layer_input_sizes) - 1
        self.dropout_rate = dropout
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.device = device
        self.layers = nn.ModuleList(
            [
                spectral_norm(nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1]))
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.gelu(layer(x))
            if self.dropout_rate > 0.0:
                x = self.dropout(x)
        # Final layer without activation
        x = self.layers[-1](x)
        return x


class ResMLP(nn.Module):
    """MLP with residual connections between layers."""

    @classmethod
    def from_config(cls, config: MLPConfig):
        return cls(
            layer_input_sizes=config.layer_input_sizes,
            dropout=config.dropout,
            device=config.device,
        )

    def __init__(
        self,
        layer_input_sizes: List[int],
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super(ResMLP, self).__init__()
        self.layer_input_sizes = layer_input_sizes
        self.num_layers = len(layer_input_sizes) - 1
        self.dropout_rate = dropout
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.device = device

        self.layers = nn.ModuleList(
            [
                spectral_norm(nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1]))
                for i in range(self.num_layers)
            ]
        )

        # Create projection layers for residual connections when dimensions don't match
        self.projections = nn.ModuleList()
        for i in range(
            self.num_layers - 1
        ):  # Exclude final layer from residual connections
            if layer_input_sizes[i] != layer_input_sizes[i + 1]:
                self.projections.append(
                    nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
                )
            else:
                self.projections.append(None)  # No projection needed

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            identity = x

            # Apply linear layer
            out = layer(x)

            # Apply residual connection
            if self.projections[i] is not None:
                identity = self.projections[i](identity)
            out = out + identity

            # Apply activation after residual
            x = F.gelu(out)

            if self.dropout_rate > 0.0:
                x = self.dropout(x)

        # Final layer without activation or residual
        x = self.layers[-1](x)
        return x


class SoftTree(nn.Module):
    """Soft Decision Tree with learnable routing through internal nodes.

    Based on "Distilling a Neural Network Into a Soft Decision Tree" (Frosst & Hinton, 2017).
    Uses sigmoid functions at internal nodes for probabilistic routing to leaf nodes.
    """

    @classmethod
    def from_config(cls, config: TreeConfig):
        return cls(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            depth=config.depth,
            sharpness=config.sharpness,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            device=config.device,
        )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 5,
        sharpness: float = 1.0,
        dropout: float = 0.0,
        batch_norm: bool = False,
        device: str = "cpu",
    ):
        super(SoftTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.sharpness = sharpness
        self.dropout_rate = dropout
        self.batch_norm_enabled = batch_norm
        self.device = device

        # Number of internal nodes and leaf nodes
        self.num_internal_nodes = 2**depth - 1
        self.num_leaf_nodes = 2**depth

        # Dropout layer
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout)

        # Internal node parameters (linear transformations for routing decisions)
        # Each internal node has a linear layer: input_dim -> 1
        self.internal_node_weights = nn.Parameter(
            torch.randn(self.num_internal_nodes, input_dim)
        )
        self.internal_node_bias = nn.Parameter(torch.randn(self.num_internal_nodes))

        # Leaf node parameters (output values)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaf_nodes, output_dim))

        # Batch normalization layer (applied to final output)
        # affine=False means no learnable weight and bias parameters
        if self.batch_norm_enabled:
            self.batch_norm = nn.BatchNorm1d(output_dim, affine=False)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        init.xavier_uniform_(self.internal_node_weights)
        init.zeros_(self.internal_node_bias)
        init.xavier_uniform_(self.leaf_weights)

    def forward(self, x):
        """Forward pass through the soft decision tree.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Handle 3D input (batch_size, 1, input_dim)
        if x.dim() == 3:
            x = x.squeeze(1)

        batch_size = x.size(0)

        # Apply dropout if enabled
        if self.dropout_rate > 0.0 and self.training:
            x = self.dropout(x)

        # Compute routing probabilities for all internal nodes
        # Shape: (batch_size, num_internal_nodes)
        routing_logits = (
            torch.matmul(x, self.internal_node_weights.t()) + self.internal_node_bias
        )
        # Apply sharpness parameter to control sigmoid steepness
        routing_probs = torch.sigmoid(self.sharpness * routing_logits)

        # Compute path probabilities to each leaf
        # We'll use dynamic programming to compute probabilities efficiently
        leaf_probs = self._compute_leaf_probabilities(routing_probs)

        # Compute weighted sum of leaf outputs
        # Shape: (batch_size, num_leaf_nodes) @ (num_leaf_nodes, output_dim) -> (batch_size, output_dim)
        output = torch.matmul(leaf_probs, self.leaf_weights)

        # Apply batch normalization if enabled
        if self.batch_norm_enabled:
            output = self.batch_norm(output)

        return output

    def _compute_leaf_probabilities(self, routing_probs):
        """Compute the probability of reaching each leaf node.

        For a binary tree:
        - Left child (0): probability = parent_prob * routing_prob
        - Right child (1): probability = parent_prob * (1 - routing_prob)

        Args:
            routing_probs: Tensor of shape (batch_size, num_internal_nodes)

        Returns:
            Tensor of shape (batch_size, num_leaf_nodes)
        """
        batch_size = routing_probs.size(0)

        # Initialize path probabilities for all nodes (internal + leaf)
        # Total nodes in complete binary tree = 2^(depth+1) - 1
        total_nodes = 2 ** (self.depth + 1) - 1
        path_probs_list = []

        # Root node has probability 1
        root_prob = torch.ones(batch_size, 1, device=self.device)
        path_probs_list.append(root_prob)

        # Traverse tree level by level to compute path probabilities
        for node_idx in range(self.num_internal_nodes):
            left_child_idx = 2 * node_idx + 1
            right_child_idx = 2 * node_idx + 2

            # Get parent probability
            parent_prob = path_probs_list[node_idx]

            # Probability of going left
            left_prob = parent_prob * routing_probs[:, node_idx : node_idx + 1]

            # Probability of going right
            right_prob = parent_prob * (1 - routing_probs[:, node_idx : node_idx + 1])

            # Append in order (left then right for binary tree structure)
            if left_child_idx < total_nodes:
                path_probs_list.append(left_prob)
            if right_child_idx < total_nodes:
                path_probs_list.append(right_prob)

        # Concatenate all path probabilities
        path_probs = torch.cat(path_probs_list, dim=1)

        # Extract leaf node probabilities (last num_leaf_nodes in the tree)
        leaf_start_idx = self.num_internal_nodes
        leaf_probs = path_probs[
            :, leaf_start_idx : leaf_start_idx + self.num_leaf_nodes
        ]

        return leaf_probs

    def compute_routing_regularization(self, x):
        """Compute Frosst & Hinton regularization for balanced routing.

        Encourages each internal node to split data 50/50 on average across the batch.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim)

        Returns:
            Regularization loss (scalar tensor)
        """
        # Handle 3D input
        if x.dim() == 3:
            x = x.squeeze(1)

        # Compute routing probabilities for all internal nodes
        # Shape: (batch_size, num_internal_nodes)
        routing_logits = (
            torch.matmul(x, self.internal_node_weights.t()) + self.internal_node_bias
        )
        routing_probs = torch.sigmoid(self.sharpness * routing_logits)

        # Compute mean routing probability for each node across the batch
        # Shape: (num_internal_nodes,)
        mean_routing_probs = routing_probs.mean(dim=0)
        # Add small epsilon for numerical stability
        eps = 1e-8
        mean_routing_probs = torch.clamp(mean_routing_probs, eps, 1 - eps)
        # Cross-entropy between (p̄, 1-p̄) and (0.5, 0.5)
        # L = -[0.5 * log(p̄) + 0.5 * log(1-p̄)]
        reg_loss = -0.5 * (
            torch.log(mean_routing_probs) + torch.log(1 - mean_routing_probs)
        )
        # Sum over all internal nodes
        return reg_loss.sum()

    def set_temperature(self, temperature: float):
        """Update sharpness parameter for temperature annealing.

        Args:
            temperature: New temperature/sharpness value
        """
        self.sharpness = temperature


class MLPEnsemble(nn.Module):
    """Ensemble of MLP or ResMLP models that averages their outputs."""

    @classmethod
    def from_config(cls, config: MLPConfig, n_ensemble, use_resmlp: bool = False):
        return cls(
            n_models=n_ensemble,
            layer_input_sizes=config.layer_input_sizes,
            dropout=config.dropout,
            device=config.device,
            use_resmlp=use_resmlp,
        )

    def __init__(
        self,
        n_models,
        layer_input_sizes,
        dropout: float = 0.0,
        device: str = "cpu",
        use_resmlp: bool = False,
    ):
        super(MLPEnsemble, self).__init__()
        model_class = ResMLP if use_resmlp else MLP
        self.models = nn.ModuleList(
            [
                model_class(
                    layer_input_sizes,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(n_models)
            ]
        )

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output


class SoftTreeEnsemble(nn.Module):
    """Ensemble of SoftTree models that averages their outputs."""

    @classmethod
    def from_config(cls, config: TreeConfig, n_ensemble: int):
        return cls(
            n_models=n_ensemble,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            depth=config.depth,
            sharpness=config.sharpness,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            device=config.device,
        )

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        output_dim: int,
        depth: int = 5,
        sharpness: float = 1.0,
        dropout: float = 0.0,
        batch_norm: bool = False,
        device: str = "cpu",
    ):
        super(SoftTreeEnsemble, self).__init__()
        self.output_dim = output_dim
        self.batch_norm_enabled = batch_norm

        self.models = nn.ModuleList(
            [
                SoftTree(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    depth=depth,
                    sharpness=sharpness,
                    dropout=dropout,
                    batch_norm=False,  # Apply batch norm after ensemble averaging
                    device=device,
                )
                for _ in range(n_models)
            ]
        )

        # Batch normalization layer (applied after ensemble averaging)
        # affine=False means no learnable weight and bias parameters
        if self.batch_norm_enabled:
            self.batch_norm = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)

        # Apply batch normalization if enabled
        if self.batch_norm_enabled:
            avg_output = self.batch_norm(avg_output)

        return avg_output

    def compute_routing_regularization(self, x):
        """Compute routing regularization for all trees in the ensemble."""
        total_reg = torch.tensor(0.0, device=x.device)
        for model in self.models:
            total_reg = total_reg + model.compute_routing_regularization(x)
        # Average across ensemble
        return total_reg / len(self.models)

    def set_temperature(self, temperature: float):
        """Update sharpness for all trees in ensemble.

        Args:
            temperature: New temperature/sharpness value
        """
        for model in self.models:
            model.set_temperature(temperature)


class VAE(nn.Module):
    """Variational Autoencoder that wraps MLP, ResMLP, or their ensembles.

    Outputs mu and logvar for variational inference. Can wrap either a single MLP/ResMLP
    or an ensemble based on n_ensemble and use_resmlp parameters.
    """

    @classmethod
    def from_config(
        cls, config: MLPConfig, n_ensemble: int = 1, use_resmlp: bool = False
    ):
        return cls(
            layer_input_sizes=config.layer_input_sizes,
            n_ensemble=n_ensemble,
            dropout=config.dropout,
            device=config.device,
            output_mu_var=config.output_mu_var,
            use_resmlp=use_resmlp,
        )

    def __init__(
        self,
        layer_input_sizes: List[int],
        n_ensemble: int = 1,
        dropout: float = 0.0,
        device: str = "cpu",
        output_mu_var: bool = False,
        use_resmlp: bool = False,
    ):
        super(VAE, self).__init__()
        self.output_mu_var = output_mu_var
        self.device = device
        self.latent_dim = layer_input_sizes[-1]

        # Double the output dimension to accommodate both mu and logvar
        modified_layer_sizes = layer_input_sizes[:-1] + [layer_input_sizes[-1] * 2]

        # Create either MLP/ResMLP or MLPEnsemble based on n_ensemble and use_resmlp
        model_class = ResMLP if use_resmlp else MLP
        if n_ensemble == 1:
            self.base_model = model_class(
                layer_input_sizes=modified_layer_sizes,
                dropout=dropout,
                device=device,
            )
        else:
            self.base_model = MLPEnsemble(
                n_models=n_ensemble,
                layer_input_sizes=modified_layer_sizes,
                dropout=dropout,
                device=device,
                use_resmlp=use_resmlp,
            )

    def reparametrization(self, mu, logvar):
        """Reparameterization trick: z = mu + exp(logvar/2) * epsilon"""
        epsilon = torch.randn_like(logvar)
        return mu + torch.exp(logvar / 2) * epsilon

    def forward(self, x):
        # Get output from base model
        out = self.base_model(x)

        # Split output into mu and logvar
        mu = out[..., : self.latent_dim]
        logvar = out[..., self.latent_dim :]

        # Evaluation mode: always return (mu, logvar)
        if not self.training:
            return mu, logvar

        # Training mode with reparameterization
        z = self.reparametrization(mu, logvar)

        if self.output_mu_var:
            return z, mu, logvar
        else:
            return z


class IndexPredictionModel(nn.Module):
    @classmethod
    def from_config(cls, config: IndexPredictionConfig):
        return cls(
            num_moderators=config.num_moderators,
            hidden_layer_sizes=config.hidden_layer_sizes,
            batch_norm=config.batch_norm,
            vae=config.vae,
            device=config.device,
            dropout=config.dropout,
            n_ensemble=config.n_ensemble,
            output_mu_var=config.output_mu_var,
            use_resmlp=config.use_resmlp,
            use_soft_tree=config.use_soft_tree,
            tree_depth=config.tree_depth,
            tree_sharpness=config.tree_sharpness,
        )

    def __init__(
        self,
        num_moderators: Union[int, Sequence[int]],
        hidden_layer_sizes: Union[Sequence[int], Sequence[Sequence[int]]],
        batch_norm: bool = True,
        vae: bool = True,
        device: str = "cpu",
        dropout: float = 0.0,
        n_ensemble: int = 1,
        output_mu_var: bool = True,
        use_resmlp: bool = False,
        use_soft_tree: bool = False,
        tree_depth: int = None,
        tree_sharpness: float = 1.0,
    ):
        super(IndexPredictionModel, self).__init__()
        self.vae = vae
        self.output_mu_var = output_mu_var
        self.device = device
        self.batch_norm = batch_norm
        self.dropout_rate = dropout
        self.n_ensemble = n_ensemble
        self.use_resmlp = use_resmlp
        self.use_soft_tree = use_soft_tree
        self.tree_depth = tree_depth
        self.tree_sharpness = tree_sharpness

        # if num_moderators is a list, then check following:
        if isinstance(num_moderators, list):
            assert isinstance(hidden_layer_sizes[0], list)
            assert len(hidden_layer_sizes) == len(num_moderators)
            self.num_models = len(num_moderators)
        else:
            self.num_models = 1

        if self.batch_norm:
            # For single model, use the last hidden layer size
            if self.num_models == 1:
                num_features = hidden_layer_sizes[-1]
                self.bn = nn.BatchNorm1d(num_features, affine=False)
            # For multiple models, sum the last hidden layer sizes
            else:
                self.bns = nn.ModuleList()
                for hs in hidden_layer_sizes:
                    num_features = hs[-1]
                    self.bns.append(nn.BatchNorm1d(num_features, affine=False))

        if self.num_models == 1:
            if use_soft_tree:
                # SoftTree backbone (VAE not supported)
                input_dim = num_moderators
                output_dim = hidden_layer_sizes[-1]

                if n_ensemble > 1:
                    self.mlp = SoftTreeEnsemble(
                        n_models=n_ensemble,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        depth=tree_depth,
                        sharpness=tree_sharpness,
                        dropout=dropout,
                        device=device,
                    )
                else:
                    self.mlp = SoftTree(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        depth=tree_depth,
                        sharpness=tree_sharpness,
                        dropout=dropout,
                        device=device,
                    )
            else:
                # MLP/ResMLP backbone
                self.num_layers = len(hidden_layer_sizes)
                layer_input_sizes = [num_moderators] + hidden_layer_sizes

                if self.vae:
                    # Use VAE wrapper which handles both single and ensemble internally
                    self.mlp = VAE(
                        layer_input_sizes=layer_input_sizes,
                        n_ensemble=n_ensemble,
                        output_mu_var=output_mu_var,
                        dropout=dropout,
                        device=device,
                        use_resmlp=use_resmlp,
                    )
                elif n_ensemble > 1:
                    self.mlp = MLPEnsemble(
                        n_ensemble,
                        layer_input_sizes,
                        dropout=dropout,
                        device=device,
                        use_resmlp=use_resmlp,
                    )
                else:
                    model_class = ResMLP if use_resmlp else MLP
                    self.mlp = model_class(
                        layer_input_sizes,
                        dropout=dropout,
                        device=device,
                    )

        else:
            # Multiple models case
            if use_soft_tree:
                # SoftTree backbone for multiple models (VAE not supported)
                self.mlp = []
                for i in range(self.num_models):
                    input_dim = num_moderators[i]
                    output_dim = hidden_layer_sizes[i][-1]

                    if n_ensemble > 1:
                        self.mlp.append(
                            SoftTreeEnsemble(
                                n_models=n_ensemble,
                                input_dim=input_dim,
                                output_dim=output_dim,
                                depth=tree_depth,
                                sharpness=tree_sharpness,
                                dropout=dropout,
                                device=device,
                            )
                        )
                    else:
                        self.mlp.append(
                            SoftTree(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                depth=tree_depth,
                                sharpness=tree_sharpness,
                                dropout=dropout,
                                device=device,
                            )
                        )
            else:
                # MLP/ResMLP backbone for multiple models
                self.num_layers = [
                    len(hidden_layer_sizes[i]) for i in range(self.num_models)
                ]
                # final layer gets added later if add_final_layer = True
                self.mlp = []
                for i in range(self.num_models):
                    layer_input_sizes = [num_moderators[i]] + hidden_layer_sizes[i]
                    if self.vae:
                        # Use VAE wrapper which handles both single and ensemble internally
                        self.mlp.append(
                            VAE(
                                layer_input_sizes=layer_input_sizes,
                                n_ensemble=n_ensemble,
                                output_mu_var=output_mu_var,
                                dropout=dropout,
                                device=device,
                                use_resmlp=use_resmlp,
                            )
                        )
                    elif n_ensemble > 1:
                        self.mlp.append(
                            MLPEnsemble(
                                n_ensemble,
                                layer_input_sizes,
                                dropout=dropout,
                                device=device,
                                use_resmlp=use_resmlp,
                            )
                        )
                    else:
                        model_class = ResMLP if use_resmlp else MLP
                        self.mlp.append(
                            model_class(
                                layer_input_sizes,
                                dropout=dropout,
                                device=device,
                            )
                        )

    def forward(self, moderators: Union[torch.Tensor, Sequence[torch.Tensor]]):
        if self.num_models == 1:
            if self.vae:
                if not self.training:
                    moderators, log_vars = self.mlp(moderators)
                else:
                    if self.output_mu_var:
                        moderators, mus, log_vars = self.mlp(moderators)
                    else:
                        moderators = self.mlp(moderators)
            else:
                moderators = self.mlp(moderators)
        else:
            outputs = [self.mlp[i](moderators[i]) for i in range(self.num_models)]
            if self.vae:
                if not self.training:
                    mods = [o[0] for o in outputs]
                    log_vars = [o[1] for o in outputs]
                else:
                    if self.output_mu_var:
                        mods = [o[0] for o in outputs]
                        mus = [o[1] for o in outputs]
                        log_vars = [o[2] for o in outputs]
                    else:
                        mods = outputs
            else:
                mods = outputs
            moderators = mods
        if self.batch_norm:
            if isinstance(moderators, list):
                # Apply batch norm to each mod separately
                normed_mods = []
                for i, mod in enumerate(moderators):
                    if mod.dim() == 1:
                        mod = torch.unsqueeze(mod, 1)
                    normed_mod = self.bns[i](mod)
                    normed_mods.append(normed_mod)
                moderators = normed_mods  # return the list of batch-normalized tensors
                predicted_index = moderators
            else:
                if moderators.dim() == 1:
                    moderators = torch.unsqueeze(moderators, 1)
                predicted_index = self.bn(moderators)
        else:
            # Keep 2D shape for predicted_index
            if isinstance(moderators, list):
                predicted_index = moderators
            else:
                predicted_index = moderators
        # Apply softplus to ensure non-negative output.
        # This guarantees the sign of the interaction is controlled purely by
        # interaction_direction in ReGNN, regardless of network initialization.
        if isinstance(predicted_index, list):
            predicted_index = [F.softplus(pi) for pi in predicted_index]
        else:
            predicted_index = F.softplus(predicted_index)

        if self.vae:
            if not self.training:
                return predicted_index, log_vars
            else:
                if self.output_mu_var:
                    return predicted_index, mus, log_vars
                else:
                    return predicted_index
        else:
            return predicted_index

    def set_temperature(self, temperature: float):
        """Update temperature for SoftTree models.

        Args:
            temperature: New temperature/sharpness value

        Note:
            Only applies to SoftTree-based models. No-op for MLP/ResMLP models.
        """
        if self.use_soft_tree:
            if self.num_models == 1:
                if hasattr(self.mlp, "set_temperature"):
                    self.mlp.set_temperature(temperature)
            else:
                # Multiple models
                for model in self.mlp:
                    if hasattr(model, "set_temperature"):
                        model.set_temperature(temperature)


class ReGNN(nn.Module):
    @classmethod
    def from_config(cls, config: ReGNNConfig):
        return cls(
            num_moderators=config.nn_config.num_moderators,
            num_controlled=config.num_controlled,
            hidden_layer_sizes=config.nn_config.hidden_layer_sizes,
            include_bias_focal_predictor=config.include_bias_focal_predictor,
            dropout=config.nn_config.dropout,
            device=config.nn_config.device,
            control_moderators=config.control_moderators,
            batch_norm=config.nn_config.batch_norm,
            vae=config.nn_config.vae,
            output_mu_var=config.nn_config.output_mu_var,
            interaction_direction=config.interaction_direction,
            n_ensemble=config.nn_config.n_ensemble,
            use_closed_form_linear_weights=config.use_closed_form_linear_weights,
            use_resmlp=config.nn_config.use_resmlp,
            use_soft_tree=config.nn_config.use_soft_tree,
            tree_depth=config.nn_config.tree_depth,
            tree_sharpness=config.nn_config.tree_sharpness,
        )

    def __init__(
        self,
        num_moderators: Union[int, Sequence[int]],
        num_controlled: int,
        hidden_layer_sizes: Union[int, Sequence[int]],
        include_bias_focal_predictor=True,
        dropout=0.5,
        device: str = "cpu",
        control_moderators: bool = False,
        batch_norm: bool = True,
        vae: bool = True,
        output_mu_var: bool = True,
        interaction_direction: str = "positive",
        n_ensemble: int = 1,
        use_closed_form_linear_weights: bool = False,
        use_resmlp: bool = False,
        use_soft_tree: bool = False,
        tree_depth: int = None,
        tree_sharpness: float = 1.0,
    ):
        super(ReGNN, self).__init__()
        self.vae = vae
        self.output_mu_var = output_mu_var
        self.num_moderators = num_moderators
        self.num_controlled = num_controlled
        self.hidden_layer_sizes = hidden_layer_sizes
        self.control_moderators = control_moderators
        self.use_closed_form_linear_weights = use_closed_form_linear_weights

        if isinstance(num_moderators, list):
            self.num_models = len(num_moderators)
        else:
            self.num_models = 1
        # controlled vars other than those included as moderators
        self.has_technical_controlled_vars = num_controlled > 0

        if control_moderators:
            if isinstance(num_moderators, list):
                num_linear = num_controlled + sum(num_moderators)
            else:
                num_linear = num_controlled + num_moderators
        else:
            num_linear = num_controlled
        # Only create controlled_var_weights if we have controlled variables
        self.has_linear_terms = num_linear > 0

        # Create interaction coefficient parameter (always created for consistency)
        self.interaction_coefficient = nn.Parameter(torch.ones(1, 1))
        
        if not self.use_closed_form_linear_weights:
            self.focal_predictor_main_weight = nn.Parameter(torch.randn(1, 1))
            # self.predicted_index_weight = nn.Parameter(torch.randn(1, self.num_models))
            self.intercept = nn.Parameter(torch.randn(1, 1))
            self.mmr_parameters = [
                self.focal_predictor_main_weight,
                self.interaction_coefficient,
                # self.predicted_index_weight,
                self.intercept,
            ]
            if self.has_linear_terms:
                self.linear_weights = nn.Linear(num_linear, 1)
                self.mmr_parameters.extend(
                    [param for param in self.linear_weights.parameters()]
                )
        else:
            # In closed-form mode, only interaction_coefficient is learned
            self.mmr_parameters = [self.interaction_coefficient]

        self.dropout_rate = dropout
        self.index_prediction_model = IndexPredictionModel(
            num_moderators,
            hidden_layer_sizes,
            batch_norm=batch_norm,
            vae=vae,
            output_mu_var=output_mu_var,
            device=device,
            dropout=dropout,
            n_ensemble=n_ensemble,
            use_resmlp=use_resmlp,
            use_soft_tree=use_soft_tree,
            tree_depth=tree_depth,
            tree_sharpness=tree_sharpness,
        )

        self.device = device
        self.batch_norm = batch_norm

        self.include_bias_focal_predictor = include_bias_focal_predictor

        if include_bias_focal_predictor:
            self.xf_bias = nn.Parameter(torch.randn(1, 1))
        self.interaction_direction = interaction_direction
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
        
        # Initialize interaction coefficient to 1.0
        if hasattr(self, 'interaction_coefficient'):
            init.ones_(self.interaction_coefficient)

    def _get_linear_term_variables(self, moderators, focal_predictor, controlled_vars):
        # Keep all tensors in 2D format (batch_size, features)
        if not self.control_moderators:
            all_linear_vars = controlled_vars
        else:
            if self.has_technical_controlled_vars and self.control_moderators:
                if self.num_models == 1:
                    all_linear_vars = torch.cat((controlled_vars, moderators), 1)
                else:
                    all_linear_vars = torch.cat((controlled_vars, *moderators), 1)
            elif not self.has_technical_controlled_vars and self.control_moderators:
                if self.num_models == 1:
                    all_linear_vars = moderators
                else:
                    all_linear_vars = torch.cat([*moderators], 1)
            elif self.has_technical_controlled_vars and not self.control_moderators:
                all_linear_vars = controlled_vars
            else:
                all_linear_vars = 0.0
        if self.use_closed_form_linear_weights:
            all_linear_vars = torch.cat([all_linear_vars, focal_predictor], 1)
        return all_linear_vars

    def forward(
        self,
        moderators,
        focal_predictor,
        controlled_predictors,
        y: Union[None, torch.tensor] = None,
        s_weights: Union[None, torch.tensor] = None,
    ):
        if self.use_closed_form_linear_weights:
            assert y is not None

        # Normalize all inputs to 2D tensors (batch_size, features)
        # Clone inputs first to avoid any in-place modification issues during backward pass
        # focal_predictor: ensure (batch_size, 1)
        if focal_predictor.ndim == 1:
            focal_predictor = focal_predictor.unsqueeze(1).clone()
        elif focal_predictor.ndim == 3:
            focal_predictor = focal_predictor.reshape(
                focal_predictor.shape[0], -1
            ).clone()
        else:
            focal_predictor = focal_predictor.clone()

        # controlled_predictors: ensure (batch_size, num_controlled)
        if controlled_predictors.ndim == 3:
            controlled_predictors = controlled_predictors.reshape(
                controlled_predictors.shape[0], -1
            ).clone()
        else:
            controlled_predictors = controlled_predictors.clone()

        # moderators: ensure 2D (batch_size, num_moderators) or list of 2D
        if not isinstance(moderators, list):
            if moderators.ndim == 3:
                moderators = moderators.reshape(moderators.shape[0], -1).clone()
            else:
                moderators = moderators.clone()
        else:
            moderators = [
                m.reshape(m.shape[0], -1).clone() if m.ndim == 3 else m.clone()
                for m in moderators
            ]

        if self.include_bias_focal_predictor:
            focal_predictor = torch.maximum(
                torch.tensor([[0.0]], device=self.device),
                (focal_predictor - torch.abs(self.xf_bias)),
            )

        # get linear term variables
        all_linear_vars = self._get_linear_term_variables(
            moderators, focal_predictor, controlled_predictors
        )  # Shape: (batch_size, n_linear_terms)

        # compute index prediction
        if self.vae:
            if not self.training:
                predicted_index, log_var = self.index_prediction_model(moderators)
            else:
                if self.output_mu_var:
                    predicted_index, mu, log_var = self.index_prediction_model(
                        moderators
                    )
                else:
                    predicted_index = self.index_prediction_model(moderators)
        else:
            predicted_index = self.index_prediction_model(moderators)

        if isinstance(predicted_index, list):
            predicted_index = torch.cat(
                predicted_index, dim=1
            )  # Shape: (batch_size, total_features)

        # compute outcome
        if not self.use_closed_form_linear_weights:
            # Calculate interaction term based on number of models
            # For multiple models, sum the predicted indices to get single interaction value
            if predicted_index.shape[1] > 1:
                predicted_interaction_term = predicted_index.sum(dim=1, keepdim=True)
            else:
                predicted_interaction_term = predicted_index
            predicted_interaction_term = predicted_interaction_term * focal_predictor

            # Apply interaction coefficient with direction constraint.
            # predicted_index is already non-negative (softplus in IndexPredictionModel),
            # so interaction_direction purely controls the sign here.
            interaction_coef = F.softplus(self.interaction_coefficient)
            if self.interaction_direction == "negative":
                interaction_coef = -interaction_coef
            
            predicted_interaction_term = interaction_coef * predicted_interaction_term

            # Only compute controlled term if we have controlled variables
            linear_terms = (
                self.linear_weights(all_linear_vars) if self.has_linear_terms else 0.0
            )

            focal_predictor_main_effect = (
                self.focal_predictor_main_weight * focal_predictor
            )

            outcome = (
                linear_terms
                + focal_predictor_main_effect
                + predicted_interaction_term
                + self.intercept
            )

        else:
            # Compute interaction term with learnable coefficient and sign constraint
            # Calculate interaction term based on number of models
            if predicted_index.shape[1] > 1:
                predicted_interaction_term = predicted_index.sum(dim=1, keepdim=True)
            else:
                predicted_interaction_term = predicted_index
            
            # Apply interaction coefficient with direction constraint.
            # predicted_index is already non-negative (softplus in IndexPredictionModel),
            # so interaction_direction purely controls the sign here.
            interaction_coef = F.softplus(self.interaction_coefficient)
            if self.interaction_direction == "negative":
                interaction_coef = -interaction_coef
            
            interaction_term = interaction_coef * predicted_interaction_term * focal_predictor  # shape: [batch, 1]
            
            # Build design matrix with only linear predictors (excluding interaction)
            # all_linear_vars already includes focal_predictor main effect
            X_linear = all_linear_vars  # shape: [batch, n_linear_terms]

            # Add intercept column (constant ones)
            ones = torch.ones(X_linear.size(0), 1, device=X_linear.device)
            X_linear = torch.cat([ones, X_linear], dim=1)  # shape: [batch, n_linear_terms + 1]

            # Adjust target by subtracting interaction term (already scaled by coefficient)
            y_adjusted = y.clone()
            
            # Apply weights if provided
            if s_weights is not None:
                sqrt_s_weights = torch.sqrt(s_weights).squeeze(-1)  # shape: [batch]
                X_linear = X_linear * sqrt_s_weights.unsqueeze(-1)  # scale each row
                y_adjusted = y_adjusted.squeeze(-1)
                y_adjusted = y_adjusted * sqrt_s_weights
                interaction_term_weighted = interaction_term.squeeze(-1) * sqrt_s_weights
            else:
                interaction_term_weighted = interaction_term.squeeze(-1)
                
            y_adjusted = torch.squeeze(y_adjusted, -1)
            
            # Subtract interaction term from targets
            y_adjusted = y_adjusted - interaction_term_weighted

            # Add small ridge regularization for numerical stability
            eps = 1e-6
            n = X_linear.size(1)
            XtX = X_linear.t() @ X_linear + eps * torch.eye(n, device=X_linear.device)
            Xty = X_linear.t() @ y_adjusted

            # Solve least squares for linear terms only
            if X_linear.size(0) < X_linear.size(1):
                # Underdetermined → pseudo-inverse
                weights = torch.linalg.pinv(X_linear) @ y_adjusted
            else:
                try:
                    L = torch.linalg.cholesky(XtX)
                    weights = torch.cholesky_solve(Xty.unsqueeze(-1), L).squeeze(-1)
                except:
                    # Fallback to SVD
                    U, S, Vh = torch.linalg.svd(X_linear, full_matrices=False)
                    threshold = eps * S.max()
                    S_inv = torch.where(
                        S > threshold, 1.0 / (S + eps), torch.zeros_like(S)
                    )
                    weights = (Vh.t() * S_inv.unsqueeze(-1)) @ (U.t() @ y_adjusted)

            # Predict outcome: linear terms + interaction term (already scaled by coefficient)
            outcome = (X_linear @ weights).view(-1, 1) + interaction_term

        if self.vae:
            if not self.training:
                return outcome
            else:
                if self.output_mu_var:
                    return outcome, mu, log_var
                else:
                    return outcome
        else:
            return outcome

    def set_temperature(self, temperature: float):
        """Update temperature for SoftTree models in index prediction.

        Args:
            temperature: New temperature/sharpness value

        Note:
            Only applies if the index_prediction_model uses SoftTree. No-op otherwise.
        """
        if hasattr(self.index_prediction_model, "set_temperature"):
            self.index_prediction_model.set_temperature(temperature)
