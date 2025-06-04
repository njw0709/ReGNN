import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Union, Sequence, List
import numpy as np
from .base import MLPConfig, IndexPredictionConfig, ReGNNConfig


class MLP(nn.Module):
    @classmethod
    def from_config(cls, config: MLPConfig):
        return cls(
            layer_input_sizes=config.layer_input_sizes,
            vae=config.vae,
            dropout=config.dropout,
            device=config.device,
            output_mu_var=config.output_mu_var,
            ensemble=config.ensemble,
        )

    def __init__(
        self,
        layer_input_sizes: List[int],
        vae: bool = False,
        dropout: float = 0.0,
        device: str = "cpu",
        output_mu_var: bool = False,
        ensemble: bool = False,
    ):
        super(MLP, self).__init__()
        self.vae = vae
        self.ensemble = ensemble
        self.layer_input_sizes = layer_input_sizes
        self.num_layers = len(layer_input_sizes) - 1
        self.output_mu_var = output_mu_var
        self.dropout_rate = dropout
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.device = device
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
                for i in range(self.num_layers - 1)
            ]
        )
        if self.vae:
            self.mu = nn.Linear(layer_input_sizes[-2], layer_input_sizes[-1])
            self.logvar = nn.Linear(layer_input_sizes[-2], layer_input_sizes[-1])
        else:
            self.output_layer = nn.Linear(layer_input_sizes[-2], layer_input_sizes[-1])

    def reparametrization(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        return mu + torch.exp(logvar / 2) * epsilon

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x))
            if self.dropout_rate > 0.0:
                x = self.dropout(x)
        if self.vae:
            mu = self.mu(x)
            logvar = self.logvar(x)
            if self.ensemble:
                return mu, logvar
            else:
                out = self.reparametrization(mu, logvar)
                if not self.training:
                    return mu, logvar
                if self.output_mu_var:
                    return out, mu, logvar
                else:
                    return out
        else:
            out = self.output_layer(x)
            return out


class MLPEnsemble(nn.Module):
    @classmethod
    def from_config(cls, config: MLPConfig, n_ensemble):
        return cls(
            n_models=n_ensemble,
            layer_input_sizes=config.layer_input_sizes,
            vae=config.vae,
            dropout=config.dropout,
            device=config.device,
            output_mu_var=config.output_mu_var,
        )

    def __init__(
        self,
        n_models,
        layer_input_sizes,
        vae: bool = False,
        dropout: float = 0.0,
        device: str = "cpu",
        output_mu_var: bool = False,
    ):
        super(MLPEnsemble, self).__init__()
        self.vae = vae
        self.output_mu_var = output_mu_var
        self.models = nn.ModuleList(
            [
                MLP(
                    layer_input_sizes,
                    vae=vae,
                    dropout=dropout,
                    device=device,
                    ensemble=True,
                )
                for _ in range(n_models)
            ]
        )

    def forward(self, x):
        if self.vae:
            mus = []
            log_vars = []
            for model in self.models:
                output = model(x)
                mus.append(output[0])
                log_vars.append(output[1])
            avg_mu = torch.mean(torch.stack(mus), dim=0)
            avg_log_vars = torch.mean(torch.stack(log_vars), dim=0)
            avg_output = model.reparametrization(avg_mu, avg_log_vars)
            if not self.training:
                return avg_mu, avg_log_vars
            else:
                if self.output_mu_var:
                    return avg_output, avg_mu, avg_log_vars
                else:
                    return avg_output
        else:
            outputs = [model(x) for model in self.models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            return avg_output


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
    ):
        super(IndexPredictionModel, self).__init__()
        self.vae = vae
        self.output_mu_var = output_mu_var
        self.device = device
        self.batch_norm = batch_norm
        self.dropout_rate = dropout
        self.n_ensemble = n_ensemble

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
            self.num_layers = len(hidden_layer_sizes)
            layer_input_sizes = [num_moderators] + hidden_layer_sizes

            if n_ensemble > 1:
                self.mlp = MLPEnsemble(
                    n_ensemble,
                    layer_input_sizes,
                    vae=self.vae,
                    output_mu_var=output_mu_var,
                    dropout=dropout,
                    device=device,
                )
            else:
                self.mlp = MLP(
                    layer_input_sizes,
                    vae=self.vae,
                    output_mu_var=output_mu_var,
                    dropout=dropout,
                    device=device,
                )

        else:
            self.num_layers = [
                len(hidden_layer_sizes[i]) for i in range(self.num_models)
            ]
            # final layer gets added later if add_final_layer = True
            self.mlp = []
            for i in range(self.num_models):
                layer_input_sizes = [num_moderators[i]] + hidden_layer_sizes[i]
                if n_ensemble > 1:
                    self.mlp.append(
                        MLPEnsemble(
                            n_ensemble,
                            layer_input_sizes,
                            vae=self.vae,
                            output_mu_var=output_mu_var,
                            dropout=dropout,
                            device=device,
                        )
                    )
                else:
                    self.mlp.append(
                        MLP(
                            layer_input_sizes,
                            vae=self.vae,
                            output_mu_var=output_mu_var,
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
            interaction_direction=config.interaction_direction,
            n_ensemble=config.nn_config.n_ensemble,
            use_closed_form_linear_weights=config.use_closed_form_linear_weights,
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
        interaction_direction: str = "positive",
        n_ensemble: int = 1,
        use_closed_form_linear_weights: bool = False,
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
        # Only create controlled_var_weights if we have controlled variables
        self.has_linear_terms = num_linear > 0

        if not self.use_closed_form_linear_weights:
            self.focal_predictor_main_weight = nn.Parameter(torch.randn(1, 1))
            self.predicted_index_weight = nn.Parameter(torch.randn(1, self.num_models))
            self.mmr_parameters = [
                self.focal_predictor_main_weight,
                self.predicted_index_weight,
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

        if focal_predictor.ndim == 2:
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
                predicted_index_weight = torch.abs(self.predicted_index_weight).reshape(
                    1, 1, self.predicted_index_weight.shape[1]
                )  # Shape: (1, 1, num_models)
                predicted_interaction_term = (
                    predicted_index_weight * predicted_index
                ).sum(
                    dim=-1, keepdim=True
                )  # Shape: (batch_size, 1, 1)
            else:
                predicted_interaction_term = (
                    torch.abs(self.predicted_index_weight) * predicted_index
                )
            predicted_interaction_term = predicted_interaction_term * focal_predictor

            if self.interaction_direction != "positive":
                predicted_interaction_term = -1.0 * predicted_interaction_term

            # Only compute controlled term if we have controlled variables
            linear_terms = (
                self.linear_weights(all_linear_vars) if self.has_linear_terms else 0.0
            )

            focal_predictor_main_effect = (
                self.focal_predictor_main_weight * focal_predictor
            )

            outcome = (
                linear_terms + focal_predictor_main_effect + predicted_interaction_term
            )

        else:
            X_full = torch.cat(
                [all_linear_vars, predicted_index * focal_predictor], dim=2
            )  # shape: [batch, 1, n_terms]
            X_full = X_full.reshape(X_full.size(0), -1)  # shape: [batch, n_terms]
            if s_weights is not None:
                sqrt_s_weights = torch.sqrt(s_weights).squeeze(-1)  # shape: [batch]
                X_full = X_full * sqrt_s_weights  # shape: [batch, n_terms]
                y = y.squeeze(-1)
                y = y * sqrt_s_weights
            y = torch.squeeze(y, -1)

            # Add small ridge regularization for numerical stability
            eps = 1e-6
            n = X_full.size(1)
            XtX = X_full.t() @ X_full + eps * torch.eye(n, device=X_full.device)
            Xty = X_full.t() @ y

            # Try Cholesky first (faster and more stable for well-conditioned matrices)
            try:
                L = torch.linalg.cholesky(XtX)
                weights = torch.cholesky_solve(Xty.unsqueeze(-1), L).squeeze(-1)
            except:
                # Fallback to SVD with explicit conditioning threshold
                U, S, Vh = torch.linalg.svd(X_full, full_matrices=False)
                # Filter small singular values
                threshold = eps * S.max()
                S_inv = torch.where(S > threshold, 1.0 / (S + eps), torch.zeros_like(S))
                weights = (Vh.t() * S_inv.unsqueeze(-1)) @ (U.t() @ y)

            # Ensure outcome has shape [batch, 1, 1] regardless of s_weights
            outcome = (X_full @ weights).view(
                -1, 1, 1
            )  # Explicitly reshape to [batch, 1, 1]

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
