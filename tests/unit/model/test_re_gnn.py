import pytest
import torch
import numpy as np
from regnn.model.base import ReGNNConfig, SVDConfig
from regnn.model.regnn import ReGNN


def test_re_gnn_creation_basic():
    config = ReGNNConfig.create(
        num_moderators=10, num_controlled=5, layer_input_sizes=[5, 1]
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


def test_re_gnn_creation_with_vae_and_output_mu_var_false():
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    print(model.vae)
    print(model.training)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)
    assert output[1].shape == (batch_size, 1)
    assert output[2].shape == (batch_size, 1)


def test_re_gnn_creation_without_vae_and_output_mu_var_false():
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    config = ReGNNConfig.create(
        num_moderators=num_moderators,
        num_controlled=5,
        layer_input_sizes=[5, 1],
        svd=SVDConfig(enabled=True, svd_matrix=svd_matrix, k_dim=k_dim),
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
    config = ReGNNConfig.create(
        num_moderators=[5, 5], num_controlled=5, layer_input_sizes=[[3, 1], [3, 1]]
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
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
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
    config = ReGNNConfig.create(
        num_moderators=10, num_controlled=5, layer_input_sizes=[5, 1], n_ensemble=3
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


def test_re_gnn_creation_with_resmlp():
    """Test ReGNN with use_resmlp=True"""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
        use_resmlp=True,
        vae=True,
        output_mu_var=True,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    
    # Verify ResMLP is being used
    from regnn.model.regnn import VAE, ResMLP
    assert isinstance(model.index_prediction_model.mlp, VAE)
    assert isinstance(model.index_prediction_model.mlp.base_model, ResMLP)
    
    # Test forward pass
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)


def test_re_gnn_creation_with_resmlp_no_vae():
    """Test ReGNN with use_resmlp=True and vae=False"""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
        use_resmlp=True,
        vae=False,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    
    # Verify ResMLP is being used directly (no VAE wrapper)
    from regnn.model.regnn import ResMLP
    assert isinstance(model.index_prediction_model.mlp, ResMLP)
    
    # Test forward pass
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)


def test_re_gnn_creation_with_resmlp_ensemble():
    """Test ReGNN with use_resmlp=True and n_ensemble>1"""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
        use_resmlp=True,
        vae=True,
        n_ensemble=3,
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    
    # Verify ResMLP ensemble is being used
    from regnn.model.regnn import VAE, MLPEnsemble, ResMLP
    assert isinstance(model.index_prediction_model.mlp, VAE)
    assert isinstance(model.index_prediction_model.mlp.base_model, MLPEnsemble)
    assert all(isinstance(m, ResMLP) for m in model.index_prediction_model.mlp.base_model.models)
    
    # Test forward pass
    model.train()
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_vars = torch.randn(batch_size, 5)
    output = model(moderators, focal_predictor, controlled_vars)
    assert output.shape == (batch_size, 1)

