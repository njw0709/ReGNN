import pytest
import torch
from regnn.config import MLPConfig
from regnn.model.mihm import MLPEnsemble


def test_mlp_ensemble_without_vae():
    config = MLPConfig(layer_input_sizes=[10, 5], ensemble=True)
    model = MLPEnsemble.from_config(config)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert output.shape == (batch_size, 5)


def test_mlp_ensemble_with_vae_not_training():
    config = MLPConfig(layer_input_sizes=[10, 5], vae=True, ensemble=True)
    model = MLPEnsemble.from_config(config)
    model.eval()
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0].shape == (batch_size, 5)
    assert output[1].shape == (batch_size, 5)


def test_mlp_ensemble_with_vae_training_output_mu_var_false():
    config = MLPConfig(
        layer_input_sizes=[10, 5], vae=True, ensemble=True, output_mu_var=False
    )
    model = MLPEnsemble.from_config(config)
    model.train()
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert output.shape == (batch_size, 5)


def test_mlp_ensemble_with_vae_training_output_mu_var_true():
    config = MLPConfig(
        layer_input_sizes=[10, 5], vae=True, ensemble=True, output_mu_var=True
    )
    model = MLPEnsemble.from_config(config)
    model.train()
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 5)
    assert output[1].shape == (batch_size, 5)
    assert output[2].shape == (batch_size, 5)
