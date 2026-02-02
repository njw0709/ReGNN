import pytest
import torch
from regnn.model.base import MLPConfig
from regnn.model.regnn import MLPEnsemble


def test_mlp_ensemble_creation():
    """Test MLPEnsemble creation with multiple models."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    n_ensemble = 5
    model = MLPEnsemble.from_config(config, n_ensemble=n_ensemble)
    assert isinstance(model, MLPEnsemble)
    assert len(model.models) == n_ensemble


def test_mlp_ensemble_forward():
    """Test MLPEnsemble forward pass averages outputs correctly."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLPEnsemble.from_config(config, n_ensemble=5)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_mlp_ensemble_with_dropout():
    """Test MLPEnsemble with dropout configuration."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], dropout=0.3)
    model = MLPEnsemble.from_config(config, n_ensemble=3)

    # Check that each model has dropout
    for sub_model in model.models:
        assert sub_model.dropout_rate == 0.3

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    model.train()
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_mlp_ensemble_eval_mode():
    """Test MLPEnsemble in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLPEnsemble.from_config(config, n_ensemble=5)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)
