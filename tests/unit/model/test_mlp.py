import pytest
import torch
from regnn.model.base import MLPConfig
from regnn.model.regnn import MLP


def test_mlp_creation():
    """Test basic MLP creation and properties."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.layer_input_sizes == [10, 5, 3]
    assert model.num_layers == 2  # Number of layer transitions (10->5, 5->3)


def test_mlp_forward():
    """Test MLP forward pass returns correct shape."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLP.from_config(config)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_mlp_with_dropout():
    """Test MLP with dropout configuration."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], dropout=0.5)
    model = MLP.from_config(config)
    assert model.dropout_rate == 0.5
    
    # Test forward pass
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    model.train()
    output = model(input_tensor)
    assert output.shape == (batch_size, 3)


def test_mlp_eval_mode():
    """Test MLP behavior in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3])
    model = MLP.from_config(config)
    model.eval()
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)
