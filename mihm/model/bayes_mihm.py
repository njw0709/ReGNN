import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class BayesMIHM(PyroModule):
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
        variational_output=False,
    ):
        super().__init__()
        controlled_var_size = controlled_var_size + interaction_var_size
        self.dropout = nn.Dropout(dropout)

        if svd:
            assert svd_matrix is not None
            assert k_dim <= interaction_var_size
            if not isinstance(svd_matrix, torch.Tensor):
                svd_matrix = torch.tensor(svd_matrix)
            self.svd_matrix = svd_matrix
            interaction_var_size = k_dim
            self.k_dim = k_dim
        self.svd = svd
        self.variational_output = variational_output
        if variational_output:
            assert hidden_layer_sizes[-1] == 2

        layer_input_sizes = [interaction_var_size] + hidden_layer_sizes
        self.num_layers = len(hidden_layer_sizes)
        self.layers = torch.nn.ModuleList(
            [
                nn.Linear(layer_input_sizes[i], layer_input_sizes[i + 1])
                for i in range(self.num_layers)
            ]
        )

        self.controlled_var_weights = PyroModule[nn.Linear](controlled_var_size, 1)
        self.controlled_var_weights.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([1, controlled_var_size]).to_event(2)
        )
        self.controlled_var_weights.bias = PyroSample(
            dist.Normal(0.0, 5.0).expand([1]).to_event(1)
        )

        if self.variational_output:
            self.interaction_weight = nn.Parameter(torch.randn(1, 1))
        else:
            self.interaction_weight = PyroSample(
                dist.HalfNormal(1.0).expand([1]).to_event(1)
            )

        self.include_interactor_bias = include_interactor_bias
        if include_interactor_bias:
            self.interactor_bias = PyroSample(
                dist.HalfNormal(1.0).expand([1]).to_event(1)
            )

    def forward(self, interaction_input_vars, interactor_var, controlled_vars, y=None):
        all_linear_vars = torch.cat((controlled_vars, interaction_input_vars), 1)
        controlled_term = self.controlled_var_weights(all_linear_vars)

        if self.svd:
            interaction_input_vars = torch.matmul(
                interaction_input_vars, self.svd_matrix[:, : self.k_dim]
            )

        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                interaction_input_vars = layer(interaction_input_vars)
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))
                interaction_input_vars = self.dropout(interaction_input_vars)

        if self.variational_output:
            mean = interaction_input_vars[:, 0]
            std = F.softplus(interaction_input_vars[:, 1])
            with pyro.plate("index_data", size=interaction_input_vars.size(0)):
                predicted_index = pyro.sample(
                    "predicted_index",
                    dist.Normal(mean, std),
                )
        else:
            predicted_index = interaction_input_vars
        if self.include_interactor_bias:
            predicted_interaction_term = (
                self.interaction_weight
                * (torch.unsqueeze(interactor_var, 1) - self.interactor_bias)
                * predicted_index
            )
        else:
            predicted_interaction_term = (
                self.interaction_weight
                * (torch.unsqueeze(interactor_var, 1))
                * predicted_index
            )

        predicted_epi = predicted_interaction_term + controlled_term
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))

        with pyro.plate("data", predicted_epi.size(0)):
            obs = pyro.sample(
                "obs", dist.Normal(predicted_epi, sigma).to_event(1), obs=y
            )
        return predicted_epi

    def get_resilience_index(self, interaction_input_vars):
        if self.svd:
            interaction_input_vars = torch.matmul(
                interaction_input_vars, self.svd_matrix[:, : self.k_dim]
            )
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                interaction_input_vars = layer(interaction_input_vars)
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))
        predicted_index = interaction_input_vars
        return predicted_index
