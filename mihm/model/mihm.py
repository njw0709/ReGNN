import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init


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
        linear_addition=False,
    ):
        super(MIHM, self).__init__()
        if linear_addition:
            controlled_var_size = controlled_var_size + interaction_var_size
        self.linear_addition = linear_addition
        if svd:
            assert svd_matrix is not None
            assert k_dim <= interaction_var_size
            if not isinstance(svd_matrix, torch.Tensor):
                svd_matrix = torch.tensor(svd_matrix)
            self.svd_matrix = svd_matrix
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
        self.controlled_var_weights = nn.Linear(controlled_var_size, 1)
        self.main_weight = nn.Parameter(torch.randn(1, 1))
        self.interaction_weight = nn.Parameter(torch.randn(1, 1))
        self.constant = nn.Parameter(torch.randn(1, 1))
        self.include_interactor_bias = include_interactor_bias
        self.dropout = nn.Dropout(dropout)
        if include_interactor_bias:
            self.interactor_bias = nn.Parameter(torch.randn(1, 1))
        self.initialize_weights()

    def forward(self, interaction_input_vars, interactor_var, controlled_vars):
        if not self.linear_addition:
            controlled_term = self.controlled_var_weights(controlled_vars)
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
            else:
                interaction_input_vars = F.gelu(layer(interaction_input_vars))
                interaction_input_vars = self.dropout(interaction_input_vars)

        predicted_index = interaction_input_vars
        if self.include_interactor_bias:
            predicted_interaction_term = (
                self.interaction_weight
                * (torch.unsqueeze(interactor_var, 1) + self.interactor_bias)
                * predicted_index
            )
        else:
            predicted_interaction_term = (
                self.interaction_weight
                * (torch.unsqueeze(interactor_var, 1))
                * predicted_index
            )
        predicted_epi = (
            self.constant
            + predicted_interaction_term
            + controlled_term
            + self.main_weight * predicted_index
        )
        return predicted_epi, predicted_index

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

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)


class IndexPredictionModel(nn.Module):
    def __init__(
        self,
        interaction_var_size,
        hidden_layer_sizes,
        svd=False,
        svd_matrix=None,
        k_dim=10,
    ):
        super(IndexPredictionModel, self).__init__()
        if svd:
            assert svd_matrix is not None
            assert k_dim <= interaction_var_size
            if not isinstance(svd_matrix, torch.Tensor):
                svd_matrix = torch.tensor(svd_matrix)
            self.svd_matrix = svd_matrix
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
        self.initialize_weights()

    def forward(self, interaction_input_vars):
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

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)


class ContPM25Model(nn.Module):
    def __init__(
        self,
        input_size,
        controlled_var_size,
        hidden_layer_sizes,
    ):
        super(ContPM25Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.layer2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.layer3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.controlled_var_weights = nn.Linear(controlled_var_size, 1)
        self.main_weight = nn.Parameter(torch.randn(1, 1))
        self.interaction_weight = nn.Parameter(torch.randn(1, 1))
        self.constant = nn.Parameter(torch.randn(1, 1))
        self.initialize_weights()

    def forward(self, interaction_vars, pm25_val, controlled_vars):
        x = self.layer1(interaction_vars)
        x = F.gelu(x)
        x = self.layer2(x)
        x = F.gelu(x)
        x = self.layer3(x)
        predicted_index = torch.tanh(x)
        predicted_interaction_term = (
            self.interaction_weight * (torch.unsqueeze(pm25_val, 1)) * predicted_index
        )
        controlled_term = self.controlled_var_weights(controlled_vars)
        predicted_epi = (
            self.constant
            + predicted_interaction_term
            + controlled_term
            + self.main_weight * predicted_index
        )
        return predicted_epi, predicted_index

    def get_resilience_index(self, interaction_vars):
        x = self.layer1(interaction_vars)
        x = F.gelu(x)
        x = self.layer2(x)
        x = F.gelu(x)
        x = self.layer3(x)
        predicted_index = torch.tanh(x)
        return predicted_index

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)


class ContHeatModel(nn.Module):
    def __init__(
        self,
        input_size,
        controlled_var_size,
        hidden_layer_sizes,
    ):
        super(ContHeatModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.layer2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.layer3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.controlled_var_weights = nn.Linear(controlled_var_size, 1)
        self.main_weight = nn.Parameter(torch.randn(1, 1))
        self.interaction_weight = nn.Parameter(torch.randn(1, 1))
        self.constant = nn.Parameter(torch.randn(1, 1))
        self.heat_bias = nn.Parameter(torch.randn(1, 1))
        self.initialize_weights()

    def forward(self, interaction_vars, heat_val, controlled_vars):
        x = self.layer1(interaction_vars)
        x = F.gelu(x)
        x = self.layer2(x)
        x = F.gelu(x)
        x = self.layer3(x)
        predicted_index = torch.tanh(x)
        predicted_interaction_term = (
            self.interaction_weight
            * (torch.unsqueeze(heat_val, 1) + self.heat_bias)
            * predicted_index
        )
        controlled_term = self.controlled_var_weights(controlled_vars)
        predicted_epi = (
            self.constant
            + predicted_interaction_term
            + controlled_term
            + self.main_weight * predicted_index
        )
        return predicted_epi, predicted_interaction_term

    def get_resilience_index(self, interaction_vars):
        x = self.layer1(interaction_vars)
        x = F.gelu(x)
        x = self.layer2(x)
        x = F.gelu(x)
        x = self.layer3(x)
        predicted_index = torch.tanh(x)
        return predicted_index

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
