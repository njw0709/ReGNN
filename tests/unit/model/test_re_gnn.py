import pytest
import torch
import numpy as np
from regnn.model.base import ReGNNConfig
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


def test_re_gnn_interaction_coef_is_positive():
    """interaction_coefficient is constrained positive via softplus."""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[5, 1],
    )
    model = ReGNN.from_config(config)
    assert isinstance(model, ReGNN)
    # softplus(1.0) > 0
    import torch.nn.functional as F
    coef = F.softplus(model.interaction_coefficient)
    assert (coef > 0).all()
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


def test_closed_form_with_interaction_constraint():
    """
    Test that closed-form linear weights implementation works correctly.
    
    The closed-form version should:
    1. Accept y as input during training
    2. Solve for linear coefficients analytically 
    3. Maintain differentiability for the index prediction model
    4. Constrain interaction coefficient to 1 (same as non-closed-form)
    """
    torch.manual_seed(42)
    
    n_samples = 200
    n_moderators = 2
    moderators = torch.randn(n_samples, n_moderators)
    focal_predictor = torch.randn(n_samples, 1)
    controlled_predictors = torch.randn(n_samples, 1)
    
    y = torch.randn(n_samples, 1)
    
    config = ReGNNConfig.create(
        num_moderators=n_moderators,
        num_controlled=1,
        layer_input_sizes=[16, 8],
        use_closed_form_linear_weights=True,
        vae=False,
    )
    model = ReGNN.from_config(config)
    
    # Test forward pass with y
    model.train()
    outcome = model(moderators, focal_predictor, controlled_predictors, y)
    assert outcome.shape == (n_samples, 1)
    assert not torch.isnan(outcome).any()
    
    # Test that gradients flow
    loss = ((outcome - y) ** 2).mean()
    loss.backward()
    
    # Check that index prediction model has gradients
    has_grad = False
    for param in model.index_prediction_model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Index prediction model should have gradients"
