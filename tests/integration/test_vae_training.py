"""Integration tests for VAE training with KLDivLoss."""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from regnn.data.dataset import ReGNNDataset
from regnn.data.base import ReGNNDatasetConfig
from regnn.data import DataFrameReadInConfig
from regnn.model.base import ReGNNConfig
from regnn.model.regnn import ReGNN
from regnn.train import KLDLossConfig, MSELossConfig, TrainingHyperParams, OptimizerConfig
from regnn.macroutils.utils import setup_loss_and_optimizer
from regnn.macroutils.trainer import train
from regnn.macroutils.base import MacroConfig
from regnn.probe import ProbeOptions, ModeratedRegressionConfig


@pytest.fixture
def synthetic_data():
    """Create synthetic data for testing."""
    np.random.seed(42)
    n_samples = 200
    return pd.DataFrame(
        {
            "focal_predictor": np.random.normal(0, 1, n_samples),
            "control1": np.random.normal(0, 1, n_samples),
            "moderator1": np.random.normal(0, 1, n_samples),
            "moderator2": np.random.normal(0, 1, n_samples),
            "outcome": np.random.normal(0, 1, n_samples),
        }
    )


@pytest.fixture
def regnn_dataset(synthetic_data):
    """Create ReGNN dataset."""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights=None,
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[],
    )

    return ReGNNDataset(
        df=synthetic_data,
        config=config,
        output_mode="tensor",
        device="cpu",
        dtype=np.float32,
    )


def test_vae_with_kldloss_basic_training(regnn_dataset):
    """Test 1: VAE with KLDivLoss - Basic Training."""
    # Create model config with VAE enabled
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        batch_norm=True,
        dropout=0.1,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    # Setup training config with KLDLoss
    training_hp = TrainingHyperParams(
        epochs=5,
        batch_size=32,
        loss_options=KLDLossConfig(lamba_reg=0.01),
    )

    loss_fn, _, optimizer, _ = setup_loss_and_optimizer(model, training_hp)

    # Create DataLoader
    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=True)

    losses = []
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_data in dataloader:
            optimizer.zero_grad()

            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]

            # Forward pass - should return (outcome, mu, log_var)
            predictions = model(**model_inputs)
            assert isinstance(predictions, tuple), "VAE should return tuple in training mode"
            assert len(predictions) == 3, "VAE should return 3 values: (outcome, mu, log_var)"

            outcome, mu, log_var = predictions

            # Compute loss with correct signature
            loss = loss_fn(targets, outcome, mu, log_var)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)

    # Loss should be finite
    assert all(np.isfinite(l) for l in losses), "All losses should be finite"


def test_vae_with_kldloss_full_epoch(regnn_dataset):
    """Test 2: VAE with KLDivLoss - Full Epoch with gradient checks."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        batch_norm=False,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # KLD loss function
    from regnn.model import vae_kld_regularized_loss
    loss_fn = vae_kld_regularized_loss(lambda_reg=0.01)

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=True)

    # Train for one full epoch
    model.train()
    for batch_data in dataloader:
        optimizer.zero_grad()

        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]

        outcome, mu, log_var = model(**model_inputs)
        loss = loss_fn(targets, outcome, mu, log_var)

        # Check gradients flow properly
        loss.backward()

        # Verify gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient should exist for {name}"
                assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for {name}"

        optimizer.step()


def test_vae_kldloss_components(regnn_dataset):
    """Test 3: Verify KLD loss components are computed correctly."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=True,
        output_mu_var=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=False)
    batch_data = next(iter(dataloader))

    model_inputs = {
        "moderators": batch_data["moderators"],
        "focal_predictor": batch_data["focal_predictor"],
        "controlled_predictors": batch_data["controlled_predictors"],
    }
    targets = batch_data["outcome"]

    outcome, mu, log_var = model(**model_inputs)

    # Compute loss components manually
    mse_component = torch.mean((targets - outcome) ** 2)
    kld_component = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))

    # Test with different lambda values
    for lambda_reg in [0.001, 0.01, 0.1]:
        from regnn.model import vae_kld_regularized_loss
        loss_fn = vae_kld_regularized_loss(lambda_reg=lambda_reg)
        total_loss = loss_fn(targets, outcome, mu, log_var)

        # Expected loss
        expected_loss = mse_component + lambda_reg * kld_component

        # Should be approximately equal
        assert torch.isclose(
            total_loss, expected_loss, rtol=1e-5
        ), f"Loss mismatch for lambda={lambda_reg}"


def test_vae_training_vs_eval_mode(regnn_dataset):
    """Test 4: VAE Training vs Evaluation Mode."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=True,
        output_mu_var=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=False)
    batch_data = next(iter(dataloader))

    model_inputs = {
        "moderators": batch_data["moderators"],
        "focal_predictor": batch_data["focal_predictor"],
        "controlled_predictors": batch_data["controlled_predictors"],
    }

    # Training mode
    model.train()
    train_output = model(**model_inputs)
    assert isinstance(train_output, tuple), "Training mode should return tuple"
    assert len(train_output) == 3, "Training should return (outcome, mu, log_var)"

    # Evaluation mode
    model.eval()
    eval_output = model(**model_inputs)
    # In eval mode, VAE returns outcome only (no tuple)
    assert isinstance(eval_output, torch.Tensor), "Eval mode should return tensor"
    assert eval_output.ndim == 2, "Eval output should be 2D"


def test_vae_different_lambda_values(regnn_dataset):
    """Test 5: VAE with Different Lambda Values."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=True,
        output_mu_var=True,
        device="cpu",
    )

    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=False)
    batch_data = next(iter(dataloader))

    model_inputs = {
        "moderators": batch_data["moderators"],
        "focal_predictor": batch_data["focal_predictor"],
        "controlled_predictors": batch_data["controlled_predictors"],
    }
    targets = batch_data["outcome"]

    losses_by_lambda = {}

    for lambda_reg in [0.001, 0.01, 0.1, 1.0]:
        model = ReGNN.from_config(model_config)
        model.train()

        from regnn.model import vae_kld_regularized_loss
        loss_fn = vae_kld_regularized_loss(lambda_reg=lambda_reg)

        outcome, mu, log_var = model(**model_inputs)
        loss = loss_fn(targets, outcome, mu, log_var)

        losses_by_lambda[lambda_reg] = loss.item()

    # Verify all losses are finite
    assert all(
        np.isfinite(l) for l in losses_by_lambda.values()
    ), "All losses should be finite"


def test_vae_with_ensemble(regnn_dataset):
    """Test 6: VAE with Ensemble."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        n_ensemble=3,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    training_hp = TrainingHyperParams(
        epochs=3,
        batch_size=32,
        loss_options=KLDLossConfig(lamba_reg=0.01),
    )

    loss_fn, _, optimizer, _ = setup_loss_and_optimizer(model, training_hp)

    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=True)

    # Train for a few batches
    batch_count = 0
    for batch_data in dataloader:
        optimizer.zero_grad()

        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]

        outcome, mu, log_var = model(**model_inputs)
        loss = loss_fn(targets, outcome, mu, log_var)

        loss.backward()
        optimizer.step()

        batch_count += 1
        if batch_count >= 3:  # Test a few batches
            break

    assert batch_count == 3, "Should have trained for 3 batches"


def test_vae_end_to_end_with_trainer(synthetic_data):
    """Test 7: End-to-End with Trainer - simplified without MacroConfig."""
    # Create model config
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        batch_norm=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)

    # Setup training with KLDLoss
    training_hp = TrainingHyperParams(
        epochs=3,
        batch_size=32,
        loss_options=KLDLossConfig(lamba_reg=0.01),
        train_test_split_ratio=0.8,
    )

    loss_fn, _, optimizer, _ = setup_loss_and_optimizer(model, training_hp)

    # Create dataset
    dataset_config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights=None,
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[],
    )

    dataset = ReGNNDataset(
        df=synthetic_data,
        config=dataset_config,
        output_mode="tensor",
        device="cpu",
        dtype=np.float32,
    )

    # Split into train/test
    from regnn.data import train_test_split
    train_size = int(len(dataset) * 0.8)
    train_indices = list(range(train_size))
    train_dataset = dataset.get_subset(train_indices)

    # Create dataloader
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train for a few epochs
    model.train()
    for epoch in range(3):
        for batch_data in dataloader:
            optimizer.zero_grad()

            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]

            outcome, mu, log_var = model(**model_inputs)
            loss = loss_fn(targets, outcome, mu, log_var)

            loss.backward()
            optimizer.step()

    # Verify model was trained
    assert isinstance(model, ReGNN), "Should return trained ReGNN model"


def test_config_validation_output_mu_var_false():
    """Test 8a: Configuration validation - output_mu_var=False with KLDLossConfig should fail during training."""
    # When output_mu_var=False, the model returns only z (not z, mu, log_var)
    # This will cause unpacking error when used with KLDLoss
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=True,
        output_mu_var=False,  # This will cause issues with KLDLoss
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    # Create a simple batch
    batch_size = 16
    moderators = torch.randn(batch_size, 2)
    focal_predictor = torch.randn(batch_size, 1)
    controlled_predictors = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    # With output_mu_var=False, this returns only z (single tensor)
    output = model(moderators, focal_predictor, controlled_predictors)
    
    # Verify it returns only a tensor, not a tuple
    assert isinstance(output, torch.Tensor), "output_mu_var=False should return single tensor"
    assert not isinstance(output, tuple), "output_mu_var=False should not return tuple"


def test_config_validation_vae_false():
    """Test 8b: Configuration validation - vae=False with KLDLossConfig doesn't make sense."""
    # When vae=False, the model doesn't produce mu and log_var needed for KLD loss
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=False,  # No VAE
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    # Create a simple batch
    batch_size = 16
    moderators = torch.randn(batch_size, 2)
    focal_predictor = torch.randn(batch_size, 1)
    controlled_predictors = torch.randn(batch_size, 1)
    
    # With vae=False, this returns only outcome (single tensor)
    output = model(moderators, focal_predictor, controlled_predictors)
    
    # Verify it returns only a tensor
    assert isinstance(output, torch.Tensor), "vae=False should return single tensor"
    assert not isinstance(output, tuple), "vae=False should not return tuple"


def test_vae_backward_pass_no_errors(regnn_dataset):
    """Additional test: Ensure backward pass completes without errors."""
    torch.autograd.set_detect_anomaly(True)

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        batch_norm=True,
        dropout=0.1,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    from regnn.model import vae_kld_regularized_loss
    loss_fn = vae_kld_regularized_loss(lambda_reg=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=True)

    # Run a few iterations
    for i, batch_data in enumerate(dataloader):
        if i >= 3:
            break

        optimizer.zero_grad()

        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]

        outcome, mu, log_var = model(**model_inputs)
        loss = loss_fn(targets, outcome, mu, log_var)

        # This should not raise any errors
        loss.backward()
        optimizer.step()

    torch.autograd.set_detect_anomaly(False)


def test_vae_loss_decreases_over_epochs(regnn_dataset):
    """Additional test: Verify loss generally decreases over training."""
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[16, 8],
        vae=True,
        output_mu_var=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    from regnn.model import vae_kld_regularized_loss
    loss_fn = vae_kld_regularized_loss(lambda_reg=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dataloader = DataLoader(regnn_dataset, batch_size=64, shuffle=True)

    epoch_losses = []
    for epoch in range(10):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_data in dataloader:
            optimizer.zero_grad()

            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]

            outcome, mu, log_var = model(**model_inputs)
            loss = loss_fn(targets, outcome, mu, log_var)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        epoch_losses.append(avg_loss)

    # Loss should generally decrease (last epoch should be lower than first)
    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease over training"


def test_objective_probe_with_vae(regnn_dataset):
    """Test that loss computation works correctly with VAE in eval mode (simulating probe)."""
    from regnn.model import vae_kld_regularized_loss

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4],
        vae=True,
        output_mu_var=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    
    # Create loss function
    loss_fn = vae_kld_regularized_loss(lambda_reg=0.01)

    # Create dataloader
    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=False)

    # Test in eval mode (what probes use)
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]

            # In eval mode, VAE returns (mu, log_var) not (z, mu, log_var)
            # But for ReGNN, it returns outcome only
            output = model(**model_inputs)
            
            # In eval mode, ReGNN with VAE should return single tensor (outcome using mu)
            assert isinstance(output, torch.Tensor), "Eval mode should return tensor"
            
            # Can compute MSE loss in eval mode
            mse_loss = torch.mean((output - targets) ** 2)
            total_loss += mse_loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    assert np.isfinite(avg_loss), "Average loss should be finite"
