import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mihm.model.modelutils import get_index_prediction_weights


class IndexPredictionModel(nn.Module):
    def __init__(
        self,
        interaction_var_size,
        hidden_layer_sizes,
        svd=False,
        svd_matrix=None,
        k_dim=10,
        batch_norm=True,
        vae=True,
        device="cpu",
    ):
        super(IndexPredictionModel, self).__init__()
        self.vae = vae
        self.device = device
        self.output_std = False
        if self.vae:
            hidden_layer_sizes[-1] = 2
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(1, affine=False)

        if svd:
            assert svd_matrix is not None
            assert k_dim <= interaction_var_size
            if not isinstance(svd_matrix, torch.Tensor):
                svd_matrix = torch.tensor(svd_matrix)
            self.svd_matrix = svd_matrix
            if device == "cuda":
                self.svd_matrix = self.svd_matrix.cuda()
            interaction_var_size = k_dim
            self.k_dim = k_dim
        self.svd = svd
        self.num_layers = len(hidden_layer_sizes)
        layer_input_sizes = [interaction_var_size] + hidden_layer_sizes
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
                for i in range(self.num_layers)
            ]
        )

    def forward(self, interaction_input_vars):
        if self.svd:
            interaction_input_vars = torch.matmul(
                interaction_input_vars, self.svd_matrix[:, : self.k_dim]
            )
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                interaction_input_vars = layer(interaction_input_vars)
                if self.vae:
                    mu, logvar = (
                        interaction_input_vars[:, 0],
                        interaction_input_vars[:, 1],
                    )
                    interaction_input_vars = mu
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))

        if self.batch_norm:
            if interaction_input_vars.dim() == 1:
                interaction_input_vars = torch.unsqueeze(interaction_input_vars, 1)
            predicted_index = self.bn(interaction_input_vars)
        else:
            predicted_index = interaction_input_vars

        if self.output_std:
            assert self.vae
            total_var = logvar + torch.log(self.bn.running_var)
            std = torch.unsqueeze(torch.exp(total_var / 2), 1)
            return std
        else:
            return predicted_index

    def get_index_mean_std(self, interaction_input_vars):
        if not self.vae:
            raise ValueError("Model is not a VAE")
        if self.svd:
            interaction_input_vars = torch.matmul(
                interaction_input_vars, self.svd_matrix[:, : self.k_dim]
            )
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                interaction_input_vars = layer(interaction_input_vars)
                mu, logvar = (
                    interaction_input_vars[:, 0],
                    interaction_input_vars[:, 1],
                )
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))

        if self.batch_norm:
            predicted_index = self.bn(torch.unsqueeze(mu, 1))
            total_var = logvar + torch.log(self.bn.running_var)
            std = torch.unsqueeze(torch.exp(total_var / 2), 1)
        else:
            predicted_index = mu
            std = torch.exp(logvar / 2)
        return predicted_index, std


class MIHM(nn.Module):
    def __init__(
        self,
        interaction_var_size,
        controlled_var_size,
        hidden_layer_sizes,
        include_interactor_bias=True,
        dropout=0.5,
        svd=False,
        svd_matrix=None,
        k_dim=10,
        device="cpu",
        concatenate_interaction_vars=False,
        batch_norm=True,
        vae=True,
    ):
        super(MIHM, self).__init__()
        self.return_logvar = False
        self.vae = vae
        if self.vae:
            hidden_layer_sizes[-1] = 2
        self.interaction_var_size = interaction_var_size
        self.controlled_var_size = controlled_var_size
        self.hidden_layer_sizes = hidden_layer_sizes

        self.concatenate_interaction_vars = concatenate_interaction_vars
        if concatenate_interaction_vars:
            controlled_var_size = controlled_var_size + interaction_var_size
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(1, affine=False)
        if svd:
            assert svd_matrix is not None
            assert k_dim <= interaction_var_size
            if not isinstance(svd_matrix, torch.Tensor):
                svd_matrix = torch.tensor(svd_matrix)
            self.svd_matrix = svd_matrix
            interaction_var_size = k_dim
            self.k_dim = k_dim
            if device == "cuda":
                self.svd_matrix = self.svd_matrix.cuda()
        self.svd = svd

        layer_input_sizes = [interaction_var_size] + hidden_layer_sizes
        self.num_layers = len(hidden_layer_sizes)

        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
                for i in range(self.num_layers)
            ]
        )
        self.controlled_var_weights = nn.Linear(controlled_var_size, 1)
        self.interactor_main_weight = nn.Parameter(torch.randn(1, 1))
        self.interaction_weight = nn.Parameter(torch.randn(1, 1))

        if not self.concatenate_interaction_vars:
            self.index_main_weight = nn.Parameter(torch.randn(1, 1))

        self.include_interactor_bias = include_interactor_bias

        if include_interactor_bias:
            self.interactor_bias = nn.Parameter(torch.randn(1, 1))

        self.initialize_weights()

    def reparametrization(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return torch.unsqueeze(mu + torch.exp(logvar / 2) * epsilon, 1)

    def forward(self, interaction_input_vars, interactor_var, controlled_vars):
        if not self.concatenate_interaction_vars:
            all_linear_vars = controlled_vars
        else:
            all_linear_vars = torch.cat((controlled_vars, interaction_input_vars), 1)
        controlled_term = self.controlled_var_weights(all_linear_vars)

        if self.svd:
            interaction_input_vars = torch.matmul(
                interaction_input_vars, self.svd_matrix[:, : self.k_dim]
            )
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                interaction_input_vars = layer(interaction_input_vars)
                if self.vae:
                    mu, logvar = (
                        interaction_input_vars[:, 0],
                        interaction_input_vars[:, 1],
                    )
                    interaction_input_vars = self.reparametrization(mu, logvar)
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))
                interaction_input_vars = self.dropout(interaction_input_vars)

        if self.batch_norm:
            predicted_index = self.bn(interaction_input_vars)
        else:
            predicted_index = interaction_input_vars

        if self.include_interactor_bias:
            interactor = torch.maximum(
                torch.tensor([[0.0]], device=self.device),
                (torch.unsqueeze(interactor_var, 1) - torch.abs(self.interactor_bias)),
            )
        else:
            interactor = torch.unsqueeze(interactor_var, 1)

        predicted_interaction_term = (
            torch.abs(self.interaction_weight) * interactor * predicted_index
        )

        interactor_main_term = self.interactor_main_weight * interactor

        if self.concatenate_interaction_vars:
            predicted_epi = (
                controlled_term + interactor_main_term + predicted_interaction_term
            )
        else:
            predicted_epi = (
                controlled_term  # does not include interaction predictors
                + interactor_main_term
                + predicted_interaction_term
                + self.index_main_weight
                * predicted_index  # interaction predictors included as main term
            )
        if self.return_logvar:
            return predicted_epi, logvar
        else:
            return predicted_epi

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)

    def get_index_prediction_model(self) -> IndexPredictionModel:
        model = IndexPredictionModel(
            self.interaction_var_size,
            self.hidden_layer_sizes,
            svd=self.svd,
            svd_matrix=self.svd_matrix,
            k_dim=self.k_dim,
            batch_norm=self.batch_norm,
            vae=self.vae,
            device=self.device,
        )

        state_dict = self.state_dict()
        index_model_state_dict = get_index_prediction_weights(state_dict)
        model.load_state_dict(index_model_state_dict)
        return model
