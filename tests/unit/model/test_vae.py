import pytest
import torch
from regnn.model.base import MLPConfig
from regnn.model.regnn import VAE


def test_vae_creation_single_mlp():
    """Test VAE creation with single MLP (n_ensemble=1)."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True)
    model = VAE.from_config(config, n_ensemble=1)
    assert isinstance(model, VAE)
    assert model.latent_dim == 3
    assert model.output_mu_var == False


def test_vae_creation_ensemble():
    """Test VAE creation with ensemble (n_ensemble>1)."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True)
    model = VAE.from_config(config, n_ensemble=5)
    assert isinstance(model, VAE)
    assert model.latent_dim == 3


def test_vae_forward_training_output_mu_var_false():
    """Test VAE forward in training mode without output_mu_var."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True, output_mu_var=False)
    model = VAE.from_config(config, n_ensemble=1)
    model.train()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    # Should return only sampled z
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)


def test_vae_forward_training_output_mu_var_true():
    """Test VAE forward in training mode with output_mu_var."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True, output_mu_var=True)
    model = VAE.from_config(config, n_ensemble=1)
    model.train()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    # Should return (z, mu, logvar)
    assert isinstance(output, tuple)
    assert len(output) == 3
    z, mu, logvar = output
    assert z.shape == (batch_size, 3)
    assert mu.shape == (batch_size, 3)
    assert logvar.shape == (batch_size, 3)


def test_vae_forward_eval_mode():
    """Test VAE forward in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True)
    model = VAE.from_config(config, n_ensemble=1)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    # Should return (mu, logvar) in eval mode
    assert isinstance(output, tuple)
    assert len(output) == 2
    mu, logvar = output
    assert mu.shape == (batch_size, 3)
    assert logvar.shape == (batch_size, 3)


def test_vae_ensemble_forward_training():
    """Test VAE with ensemble in training mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True, output_mu_var=True)
    model = VAE.from_config(config, n_ensemble=5)
    model.train()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    # Should return (z, mu, logvar)
    assert isinstance(output, tuple)
    assert len(output) == 3
    z, mu, logvar = output
    assert z.shape == (batch_size, 3)
    assert mu.shape == (batch_size, 3)
    assert logvar.shape == (batch_size, 3)


def test_vae_ensemble_forward_eval():
    """Test VAE with ensemble in eval mode."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True)
    model = VAE.from_config(config, n_ensemble=5)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    # Should return (mu, logvar) in eval mode
    assert isinstance(output, tuple)
    assert len(output) == 2
    mu, logvar = output
    assert mu.shape == (batch_size, 3)
    assert logvar.shape == (batch_size, 3)


def test_vae_reparametrization():
    """Test VAE reparameterization trick."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True)
    model = VAE.from_config(config, n_ensemble=1)

    batch_size = 32
    mu = torch.zeros(batch_size, 3)
    logvar = torch.zeros(batch_size, 3)

    # Test reparameterization
    z = model.reparametrization(mu, logvar)
    assert z.shape == (batch_size, 3)

    # With logvar=0, z should be close to mu (since exp(0/2)=1)
    # but with some noise from epsilon


def test_vae_with_dropout():
    """Test VAE with dropout."""
    config = MLPConfig(layer_input_sizes=[10, 5, 3], vae=True, dropout=0.5)
    model = VAE.from_config(config, n_ensemble=1)
    model.train()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 3)
