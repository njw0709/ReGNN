import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Union, Sequence, List
import numpy as np
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
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
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
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
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
            device=config.device,
        )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 5,
        sharpness: float = 1.0,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super(SoftTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.sharpness = sharpness
        self.dropout_rate = dropout
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
        path_probs = torch.zeros(batch_size, total_nodes, device=self.device)

        # Root node has probability 1
        path_probs[:, 0] = 1.0

        # Traverse tree level by level to compute path probabilities
        for node_idx in range(self.num_internal_nodes):
            left_child_idx = 2 * node_idx + 1
            right_child_idx = 2 * node_idx + 2

            # Probability of going left
            path_probs[:, left_child_idx] = (
                path_probs[:, node_idx] * routing_probs[:, node_idx]
            )
            # Probability of going right
            path_probs[:, right_child_idx] = path_probs[:, node_idx] * (
                1 - routing_probs[:, node_idx]
            )

        # Extract leaf node probabilities (last num_leaf_nodes in the tree)
        leaf_start_idx = self.num_internal_nodes
        leaf_probs = path_probs[
            :, leaf_start_idx : leaf_start_idx + self.num_leaf_nodes
        ]

        return leaf_probs


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
        device: str = "cpu",
    ):
        super(SoftTreeEnsemble, self).__init__()
        self.models = nn.ModuleList(
            [
                SoftTree(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    depth=depth,
                    sharpness=sharpness,
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
            svd=config.svd.enabled,
            svd_matrix=config.svd.svd_matrix,
            k_dim=config.svd.k_dim,
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
        svd: bool = False,
        svd_matrix: Union[
            None, torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]
        ] = None,
        k_dim: Union[None, int, Sequence[int]] = 10,
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
            if svd:
                assert isinstance(svd_matrix, list)
                assert len(svd_matrix) == len(num_moderators)
                assert isinstance(k_dim, list)
                assert len(k_dim) == len(num_moderators)
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

        if svd:
            assert svd_matrix is not None
            if isinstance(num_moderators, list):
                for i in range(self.num_models):
                    assert k_dim[i] <= num_moderators[i]
                    num_moderators[i] = k_dim[i]
                if not isinstance(svd_matrix[0], torch.Tensor):
                    if device == "cuda":
                        svd_matrix = [torch.Tensor(m).cuda() for m in svd_matrix]
                    else:
                        svd_matrix = [torch.Tensor(m) for m in svd_matrix]
            else:
                assert k_dim <= num_moderators
                if not isinstance(svd_matrix, torch.Tensor):
                    svd_matrix = torch.tensor(svd_matrix)
                if device == "cuda":
                    svd_matrix = svd_matrix.cuda()
                num_moderators = k_dim
            self.svd_matrix = svd_matrix
            self.k_dim = k_dim
        self.svd = svd

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
            if self.svd:
                moderators = torch.matmul(moderators, self.svd_matrix[:, : self.k_dim])
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
            if self.svd:
                moderators = [
                    torch.matmul(moderators[i], self.svd_matrix[i][:, : self.k_dim[i]])
                    for i in range(self.num_models)
                ]

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
                    normed_mods.append(self.bns[i](mod))
                moderators = normed_mods  # return the list of batch-normalized tensors
                predicted_index = moderators
            else:
                if moderators.dim() == 1:
                    moderators = torch.unsqueeze(moderators, 1)
                predicted_index = self.bn(moderators)
        else:
            # Ensure proper shape for predicted_index (batch, 1, features)
            if isinstance(moderators, list):
                predicted_index = moderators
            else:
                if moderators.dim() == 2:
                    # Add middle dimension for consistency
                    predicted_index = moderators.unsqueeze(1)
                else:
                    predicted_index = moderators
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


class ReGNN(nn.Module):
    @classmethod
    def from_config(cls, config: ReGNNConfig):
        return cls(
            num_moderators=config.nn_config.num_moderators,
            num_controlled=config.num_controlled,
            hidden_layer_sizes=config.nn_config.hidden_layer_sizes,
            include_bias_focal_predictor=config.include_bias_focal_predictor,
            dropout=config.nn_config.dropout,
            svd=config.nn_config.svd.enabled,
            svd_matrix=config.nn_config.svd.svd_matrix,
            k_dim=config.nn_config.svd.k_dim,
            device=config.nn_config.device,
            control_moderators=config.control_moderators,
            batch_norm=config.nn_config.batch_norm,
            vae=config.nn_config.vae,
            output_mu_var=config.nn_config.output_mu_var,
            # interaction_direction=config.interaction_direction,
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
        svd: bool = False,
        svd_matrix: Union[
            None, torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]
        ] = None,
        k_dim: Union[None, int, Sequence[int]] = 10,
        device: str = "cpu",
        control_moderators: bool = False,
        batch_norm: bool = True,
        vae: bool = True,
        output_mu_var: bool = True,
        # interaction_direction: str = "positive",
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

        if not self.use_closed_form_linear_weights:
            self.focal_predictor_main_weight = nn.Parameter(torch.randn(1, 1))
            # self.predicted_index_weight = nn.Parameter(torch.randn(1, self.num_models))
            self.intercept = nn.Parameter(torch.randn(1, 1))
            self.mmr_parameters = [
                self.focal_predictor_main_weight,
                # self.predicted_index_weight,
                self.intercept,
            ]
            if self.has_linear_terms:
                self.linear_weights = nn.Linear(num_linear, 1)
                self.mmr_parameters.extend(
                    [param for param in self.linear_weights.parameters()]
                )
        else:
            self.mmr_parameters = []

        self.dropout_rate = dropout
        self.index_prediction_model = IndexPredictionModel(
            num_moderators,
            hidden_layer_sizes,
            svd=svd,
            svd_matrix=svd_matrix,
            k_dim=k_dim,
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
        # self.interaction_direction = interaction_direction
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)

    def _get_linear_term_variables(self, moderators, focal_predictor, controlled_vars):
        if not self.control_moderators:
            all_linear_vars = controlled_vars
        else:
            if self.has_technical_controlled_vars and self.control_moderators:
                if self.num_models == 1:
                    all_linear_vars = torch.cat((controlled_vars, moderators), 2)
                else:
                    all_linear_vars = torch.cat((controlled_vars, *moderators), 2)
            elif not self.has_technical_controlled_vars and self.control_moderators:
                if self.num_models == 1:
                    all_linear_vars = moderators
                else:
                    all_linear_vars = torch.cat([*moderators], 2)
            elif self.has_technical_controlled_vars and not self.control_moderators:
                all_linear_vars = controlled_vars
            else:
                all_linear_vars = 0.0
        if self.use_closed_form_linear_weights:
            all_linear_vars = torch.cat([all_linear_vars, focal_predictor], 2)
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
        # shapes: moderators: (batch, 1, n); focal_predictor: (batch, 1, 1);
        # threshold focal predictor if indicated.

        if focal_predictor.ndim == 1:
            focal_predictor = torch.unsqueeze(focal_predictor, 1)

        if self.include_bias_focal_predictor:
            focal_predictor = torch.maximum(
                torch.tensor([[0.0]], device=self.device),
                (focal_predictor - torch.abs(self.xf_bias)),
            )

        # get linear term variables
        all_linear_vars = self._get_linear_term_variables(
            moderators, focal_predictor, controlled_predictors
        )  # Shape: (batch_size, 1, n_linear_terms)

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
                predicted_index, dim=2
            )  # Shape: (batch_size, 1, num_models)

        # compute outcome
        if not self.use_closed_form_linear_weights:
            # Calculate interaction term based on number of models
            if isinstance(predicted_index, list):
                predicted_interaction_term = predicted_index.sum(
                    dim=-1, keepdim=True
                )  # Shape: (batch_size, 1, 1)
            else:
                predicted_interaction_term = predicted_index
            predicted_interaction_term = predicted_interaction_term * focal_predictor

            # if self.interaction_direction != "positive":
            #     predicted_interaction_term = -1.0 * predicted_interaction_term

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
            # Build design matrix with predictors and interaction
            X_full = torch.cat(
                [all_linear_vars, predicted_index * focal_predictor], dim=2
            )  # shape: [batch, 1, n_terms]
            X_full = X_full.reshape(X_full.size(0), -1)  # shape: [batch, n_terms]

            # Add intercept column (constant ones)
            ones = torch.ones(X_full.size(0), 1, device=X_full.device)
            X_full = torch.cat([ones, X_full], dim=1)  # shape: [batch, n_terms + 1]

            # Apply weights if provided
            if s_weights is not None:
                sqrt_s_weights = torch.sqrt(s_weights).squeeze(-1)  # shape: [batch]
                X_full = X_full * sqrt_s_weights.unsqueeze(-1)  # scale each row
                y = y.squeeze(-1)
                y = y * sqrt_s_weights
            y = torch.squeeze(y, -1)

            # Add small ridge regularization for numerical stability
            eps = 1e-6
            n = X_full.size(1)
            XtX = X_full.t() @ X_full + eps * torch.eye(n, device=X_full.device)
            Xty = X_full.t() @ y

            # Solve least squares
            if X_full.size(0) < X_full.size(1):
                # Underdetermined → pseudo-inverse
                weights = torch.linalg.pinv(X_full) @ y
            else:
                try:
                    L = torch.linalg.cholesky(XtX)
                    weights = torch.cholesky_solve(Xty.unsqueeze(-1), L).squeeze(-1)
                except:
                    # Fallback to SVD
                    U, S, Vh = torch.linalg.svd(X_full, full_matrices=False)
                    threshold = eps * S.max()
                    S_inv = torch.where(
                        S > threshold, 1.0 / (S + eps), torch.zeros_like(S)
                    )
                    weights = (Vh.t() * S_inv.unsqueeze(-1)) @ (U.t() @ y)

            # Predict outcome (keep consistent shape)
            outcome = (X_full @ weights).view(-1, 1, 1)

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
