import pytest
import torch
from regnn.model.base import MLPConfig, IndexPredictionConfig
from regnn.model.regnn import ResMLP, MLPEnsemble


def test_resmlp_creation():
    """Test basic ResMLP creation and properties."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = ResMLP.from_config(config)
    assert isinstance(model, ResMLP)
    assert model.layer_input_sizes == [10, 5, 3]
    assert model.num_layers == 2  # Number of layer transitions (10->5, 5->3)


def test_resmlp_forward():
    """Test ResMLP forward pass returns correct shape."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = ResMLP.from_config(config)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_with_dropout():
    """Test ResMLP with dropout configuration."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], dropout=0.5)
    model = ResMLP.from_config(config)
    assert model.dropout_rate == 0.5
    
    # Test forward pass
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    model.train()
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_eval_mode():
    """Test ResMLP behavior in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = ResMLP.from_config(config)
    model.eval()
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_with_matching_dimensions():
    """Test ResMLP with matching dimensions (no projection needed)."""
    config = MLPConfig(layer_input_sizes=[10, 10, 10])
    model = ResMLP.from_config(config)
    
    # All projections should be None for matching dimensions
    assert all(proj is None for proj in model.projections)
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10)


def test_resmlp_with_mismatched_dimensions():
    """Test ResMLP with mismatched dimensions (projection needed)."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = ResMLP.from_config(config)
    
    # Projections should exist for dimension mismatches
    assert model.projections[0] is not None  # 10 -> 5 needs projection
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_ensemble_creation():
    """Test MLPEnsemble creation with ResMLP models."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    n_ensemble = 5
    model = MLPEnsemble.from_config(config, n_ensemble=n_ensemble, use_resmlp=True)
    assert isinstance(model, MLPEnsemble)
    assert len(model.models) == n_ensemble
    # Verify that all models are ResMLP instances
    assert all(isinstance(m, ResMLP) for m in model.models)


def test_resmlp_ensemble_forward():
    """Test MLPEnsemble with ResMLP forward pass averages outputs correctly."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLPEnsemble.from_config(config, n_ensemble=5, use_resmlp=True)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_ensemble_with_dropout():
    """Test MLPEnsemble with ResMLP and dropout configuration."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], dropout=0.3)
    model = MLPEnsemble.from_config(config, n_ensemble=3, use_resmlp=True)

    # Check that each model has dropout
    for sub_model in model.models:
        assert isinstance(sub_model, ResMLP)
        assert sub_model.dropout_rate == 0.3

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    model.train()
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_resmlp_ensemble_eval_mode():
    """Test MLPEnsemble with ResMLP in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLPEnsemble.from_config(config, n_ensemble=5, use_resmlp=True)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_index_prediction_model_with_resmlp():
    """Test IndexPredictionModel creates ResMLP when use_resmlp=True."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5],
        vae=False,
        use_resmlp=True,
    )
    from regnn.model.regnn import IndexPredictionModel
    model = IndexPredictionModel.from_config(config)
    
    # Verify the underlying MLP is a ResMLP
    assert isinstance(model.mlp, ResMLP)
    
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    output = model(moderators)
    assert output.shape == (batch_size, 1)


def test_index_prediction_model_with_resmlp_ensemble():
    """Test IndexPredictionModel creates ResMLP ensemble when use_resmlp=True and n_ensemble>1."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5],
        vae=False,
        use_resmlp=True,
        n_ensemble=3,
    )
    from regnn.model.regnn import IndexPredictionModel
    model = IndexPredictionModel.from_config(config)
    
    # Verify the underlying MLP is an ensemble of ResMLP
    assert isinstance(model.mlp, MLPEnsemble)
    assert all(isinstance(m, ResMLP) for m in model.mlp.models)
    
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    output = model(moderators)
    assert output.shape == (batch_size, 1)


def test_index_prediction_model_with_resmlp_and_vae():
    """Test IndexPredictionModel with ResMLP and VAE wrapper."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5, 2],
        vae=True,
        use_resmlp=True,
        output_mu_var=True,
    )
    from regnn.model.regnn import IndexPredictionModel, VAE
    model = IndexPredictionModel.from_config(config)
    
    # Verify the underlying model is a VAE
    assert isinstance(model.mlp, VAE)
    # Verify VAE wraps ResMLP
    assert isinstance(model.mlp.base_model, ResMLP)
    
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    model.train()
    output = model(moderators)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)
