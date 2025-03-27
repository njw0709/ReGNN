import pytest
import torch
import numpy as np
from regnn.model.base import ReGNNConfig
from regnn.model.regnn import ReGNN


def test_re_gnn_creation_basic():
    config = ReGNNConfig(num_moderators=10, num_controlled=5, hidden_layer_sizes=[5, 1])
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_vae_and_output_mu_var_false():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        vae=True,
        output_mu_var=False,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_vae_and_output_mu_var_true():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        vae=True,
        output_mu_var=True,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)
    assert output[1].shape == (batch_size, 1)
    assert output[2].shape == (batch_size, 1)


def test_re_gnn_creation_without_vae_and_output_mu_var_false():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        vae=False,
        output_mu_var=False,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_svd():
    num_moderators = 10
    k_dim = 5
    svd_matrix = np.random.rand(num_moderators, num_moderators).astype(np.float32)
    config = ReGNNConfig(
        num_moderators=num_moderators,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        svd=True,
        svd_matrix=svd_matrix,
        k_dim=k_dim,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, num_moderators)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_multiple_moderators():
    config = ReGNNConfig(
        num_moderators=[5, 5], num_controlled=5, hidden_layer_sizes=[[3, 1], [3, 1]]
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators1 = torch.randn(batch_size, 5)
    moderators2 = torch.randn(batch_size, 5)
    moderators = [moderators1, moderators2]
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_control_moderators():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        control_moderators=True,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_different_interaction_directions():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        interaction_direction="negative",
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_without_bias_for_focal_predictor():
    config = ReGNNConfig(
        num_moderators=10,
        num_controlled=5,
        hidden_layer_sizes=[5, 1],
        include_bias_focal_predictor=False,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_ensemble():
    config = ReGNNConfig(
        num_moderators=10, num_controlled=5, hidden_layer_sizes=[5, 1], n_ensemble=3
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)
