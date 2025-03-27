import pytest
import torch
from regnn.model.base import MLPConfig
from regnn.model.regnn import MLP


def test_mlp_creation_without_vae():
    config = MLPConfig(layer_input_sizes=[10, 5])
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.vae == False
    assert model.output_mu_var == False
    assert model.ensemble == False


def test_mlp_creation_with_vae():
    config = MLPConfig(layer_input_sizes=[10, 5], vae=True)
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.vae == True
    assert model.output_mu_var == False
    assert model.ensemble == False


def test_mlp_output_mu_var():
    config = MLPConfig(layer_input_sizes=[10, 5], vae=True, output_mu_var=True)
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.vae == True
    assert model.output_mu_var == True
    assert model.ensemble == False

    # Test forward pass
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, tuple)
    assert len(output) == 3  # Expecting (output, mu, log_var)
    assert output[0].shape == (batch_size, 5)
    assert output[1].shape == (batch_size, 5)
    assert output[2].shape == (batch_size, 5)


def test_mlp_ensemble_creation():
    config = MLPConfig(layer_input_sizes=[10, 5], ensemble=True)
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.vae == False
    assert model.output_mu_var == False
    assert model.ensemble == True


def test_mlp_ensemble_output_mu_var():
    config = MLPConfig(
        layer_input_sizes=[10, 5], vae=True, output_mu_var=True, ensemble=True
    )
    model = MLP.from_config(config)
    assert isinstance(model, MLP)
    assert model.vae == True
    assert model.output_mu_var == True
    assert model.ensemble == True

    # Test forward pass
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, tuple)
    assert len(output) == 2  # Expecting (mu, log_var)
    assert output[0].shape == (batch_size, 5)
    assert output[1].shape == (batch_size, 5)
