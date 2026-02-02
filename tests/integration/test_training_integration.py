"""Integration tests for ReGNN training with actual DataLoader and backward passes."""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from regnn.data.dataset import ReGNNDataset
from regnn.data.base import ReGNNDatasetConfig
from regnn.model.base import ReGNNConfig
from regnn.model.regnn import ReGNN


@pytest.fixture
def synthetic_data():
    """Create synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
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


@pytest.mark.parametrize(
    "use_soft_tree,tree_depth,vae,batch_norm",
    [
        (False, None, True, True),  # MLP with VAE and batch norm
        (False, None, False, True),  # MLP without VAE, with batch norm
        (False, None, False, False),  # MLP without VAE or batch norm
        (True, 3, False, False),  # SoftTree (VAE not supported)
        (True, 3, False, True),  # SoftTree with batch norm
    ],
)
def test_training_backward_pass(
    regnn_dataset, use_soft_tree, tree_depth, vae, batch_norm
):
    """Test full training loop with backward pass for different configurations."""

    # Enable anomaly detection for SoftTree tests to find in-place operations
    if use_soft_tree:
        torch.autograd.set_detect_anomaly(True)

    # Create model config
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4] if not use_soft_tree else [8],
        use_soft_tree=use_soft_tree,
        tree_depth=tree_depth,
        vae=vae,
        batch_norm=batch_norm,
        dropout=0.0,
        device="cpu",
        output_mu_var=False,
    )

    model = ReGNN.from_config(model_config)
    model.train()

    # Create DataLoader
    dataloader = DataLoader(
        regnn_dataset,
        batch_size=16,
        shuffle=True,
    )

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training iteration
    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        # Prepare model inputs
        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]

        # Forward pass
        if vae and model.training:
            # VAE returns (prediction, mu, logvar) in training mode with output_mu_var=False
            predictions = model(**model_inputs)
        else:
            predictions = model(**model_inputs)

        # Loss computation
        loss = loss_fn(predictions, targets)

        # Backward pass - THIS IS WHERE IN-PLACE ERRORS OCCUR
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Only test first batch
        break

    # If we get here without RuntimeError, test passes
    assert True


def test_training_with_3d_inputs(regnn_dataset):
    """Test that model handles 3D inputs from DataLoader correctly."""

    # Create model
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=False,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    # Create DataLoader
    dataloader = DataLoader(regnn_dataset, batch_size=16)

    # Get one batch
    batch_data = next(iter(dataloader))

    # Manually add extra dimensions to simulate potential 3D input issue
    model_inputs = {
        "moderators": batch_data["moderators"].unsqueeze(1),  # Make 3D
        "focal_predictor": batch_data["focal_predictor"].unsqueeze(1),  # Make 2D/3D
        "controlled_predictors": batch_data["controlled_predictors"].unsqueeze(
            1
        ),  # Make 3D
    }

    # Forward pass should handle 3D inputs
    predictions = model(**model_inputs)

    # Check output shape is 2D
    assert predictions.ndim == 2
    assert predictions.shape == (16, 1)


def test_training_shape_consistency(regnn_dataset):
    """Test that model output matches target shape through full training loop."""

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    model.train()

    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=False)

    for batch_data in dataloader:
        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]

        predictions = model(**model_inputs)

        # Check shapes match for loss computation
        assert (
            predictions.shape[0] == targets.shape[0]
        ), f"Batch size mismatch: predictions {predictions.shape} vs targets {targets.shape}"

        # Predictions should be 2D: (batch, 1)
        assert (
            predictions.ndim == 2
        ), f"Predictions should be 2D, got {predictions.ndim}D with shape {predictions.shape}"

        # Can compute loss without shape mismatch
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, targets)

        # Loss should be scalar
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"

        break  # Test first batch only


def test_multiple_epochs_no_error(regnn_dataset):
    """Test multiple epochs to ensure no accumulated errors."""

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=True,
        device="cpu",
    )

    model = ReGNN.from_config(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=True)

    # Run multiple epochs
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0

        for batch_data in dataloader:
            optimizer.zero_grad()

            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]

            predictions = model(**model_inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Should complete without errors
        assert epoch_loss > 0


# ============================================================================
# Scheduler and Temperature Annealing Integration Tests
# ============================================================================


def test_training_with_step_lr_scheduler(regnn_dataset):
    """Test training with StepLR scheduler."""
    from regnn.train import StepLRConfig, MSELossConfig
    from regnn.macroutils.utils import setup_loss_and_optimizer
    from regnn.train import TrainingHyperParams, OptimizerConfig

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=False,
        batch_norm=False,
        device="cpu",
    )
    model = ReGNN.from_config(model_config)

    # Setup training config with scheduler
    scheduler_config = StepLRConfig(step_size=2, gamma=0.5)
    optimizer_config = OptimizerConfig(scheduler=scheduler_config)
    training_hp = TrainingHyperParams(
        epochs=5,
        batch_size=16,
        optimizer_config=optimizer_config,
        loss_options=MSELossConfig(),
    )

    loss_fn, _, optimizer, scheduler = setup_loss_and_optimizer(model, training_hp)

    assert scheduler is not None
    initial_lr = optimizer.param_groups[0]["lr"]

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=True)

    for epoch in range(5):
        for batch_data in dataloader:
            optimizer.zero_grad()
            model_inputs = {
                k: batch_data[k]
                for k in ["moderators", "focal_predictor", "controlled_predictors"]
            }
            predictions = model(**model_inputs)
            loss = loss_fn(predictions, batch_data["outcome"])
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Check LR changes according to schedule
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch == 0:
            assert current_lr == initial_lr
        elif epoch == 2:
            # After 2 steps (at epoch 2), LR should be reduced
            assert current_lr < initial_lr


def test_training_with_cosine_lr_scheduler(regnn_dataset):
    """Test training with CosineAnnealingLR scheduler."""
    from regnn.train import CosineAnnealingLRConfig, MSELossConfig
    from regnn.macroutils.utils import setup_loss_and_optimizer
    from regnn.train import TrainingHyperParams, OptimizerConfig

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        vae=False,
        batch_norm=False,
        device="cpu",
    )
    model = ReGNN.from_config(model_config)

    scheduler_config = CosineAnnealingLRConfig(T_max=5, eta_min=0.0)
    optimizer_config = OptimizerConfig(scheduler=scheduler_config)
    training_hp = TrainingHyperParams(
        epochs=5,
        batch_size=16,
        optimizer_config=optimizer_config,
        loss_options=MSELossConfig(),
    )

    loss_fn, _, optimizer, scheduler = setup_loss_and_optimizer(model, training_hp)

    assert scheduler is not None
    initial_lr = optimizer.param_groups[0]["lr"]

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=False)

    lrs = []
    for epoch in range(5):
        for batch_data in dataloader:
            optimizer.zero_grad()
            model_inputs = {
                k: batch_data[k]
                for k in ["moderators", "focal_predictor", "controlled_predictors"]
            }
            predictions = model(**model_inputs)
            loss = loss_fn(predictions, batch_data["outcome"])
            loss.backward()
            optimizer.step()

        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # LR should decrease then increase (cosine)
    assert lrs[0] > lrs[2]  # Middle should be lower


def test_training_with_temperature_annealing(regnn_dataset):
    """Test training with temperature annealing for SoftTree."""
    from regnn.train import TemperatureAnnealingConfig, MSELossConfig
    from regnn.macroutils.utils import setup_loss_and_optimizer, TemperatureAnnealer
    from regnn.train import TrainingHyperParams, OptimizerConfig

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=1.0,
        vae=False,
        batch_norm=False,
        device="cpu",
    )
    model = ReGNN.from_config(model_config)

    temp_config = TemperatureAnnealingConfig(
        schedule_type="linear", initial_temp=1.0, final_temp=5.0
    )
    optimizer_config = OptimizerConfig(temperature_annealing=temp_config)
    training_hp = TrainingHyperParams(
        epochs=5,
        batch_size=16,
        optimizer_config=optimizer_config,
        loss_options=MSELossConfig(),
    )

    loss_fn, _, optimizer, scheduler = setup_loss_and_optimizer(model, training_hp)

    # Create temperature annealer
    annealer = TemperatureAnnealer(temp_config, training_hp.epochs)

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=False)

    temperatures = []
    for epoch in range(5):
        # Update temperature at start of epoch
        new_temp = annealer.update_model_temperature(model, epoch)
        temperatures.append(new_temp)

        for batch_data in dataloader:
            optimizer.zero_grad()
            model_inputs = {
                k: batch_data[k]
                for k in ["moderators", "focal_predictor", "controlled_predictors"]
            }
            predictions = model(**model_inputs)
            loss = loss_fn(predictions, batch_data["outcome"])
            loss.backward()
            optimizer.step()

    # Temperature should increase linearly
    assert temperatures[0] == 1.0
    assert temperatures[-1] == 5.0
    assert all(
        temperatures[i] < temperatures[i + 1] for i in range(len(temperatures) - 1)
    )
    # Verify model sharpness was updated
    assert model.index_prediction_model.mlp.sharpness == 5.0


def test_training_with_scheduler_and_temperature_annealing(regnn_dataset):
    """Test training with both LR scheduler and temperature annealing."""
    from regnn.train import StepLRConfig, TemperatureAnnealingConfig, MSELossConfig
    from regnn.macroutils.utils import setup_loss_and_optimizer, TemperatureAnnealer
    from regnn.train import TrainingHyperParams, OptimizerConfig

    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=1.0,
        vae=False,
        batch_norm=False,
        device="cpu",
    )
    model = ReGNN.from_config(model_config)

    scheduler_config = StepLRConfig(step_size=2, gamma=0.5)
    temp_config = TemperatureAnnealingConfig(
        schedule_type="linear", initial_temp=1.0, final_temp=5.0
    )
    optimizer_config = OptimizerConfig(
        scheduler=scheduler_config, temperature_annealing=temp_config
    )
    training_hp = TrainingHyperParams(
        epochs=5,
        batch_size=16,
        optimizer_config=optimizer_config,
        loss_options=MSELossConfig(),
    )

    loss_fn, _, optimizer, scheduler = setup_loss_and_optimizer(model, training_hp)

    assert scheduler is not None
    annealer = TemperatureAnnealer(temp_config, training_hp.epochs)

    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=False)

    initial_lr = optimizer.param_groups[0]["lr"]
    lrs = []
    temps = []

    for epoch in range(5):
        # Update temperature
        new_temp = annealer.update_model_temperature(model, epoch)
        temps.append(new_temp)

        for batch_data in dataloader:
            optimizer.zero_grad()
            model_inputs = {
                k: batch_data[k]
                for k in ["moderators", "focal_predictor", "controlled_predictors"]
            }
            predictions = model(**model_inputs)
            loss = loss_fn(predictions, batch_data["outcome"])
            loss.backward()
            optimizer.step()

        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Both should have changed
    assert temps[0] < temps[-1]  # Temperature increased
    assert lrs[0] == initial_lr
    assert lrs[2] < initial_lr  # LR reduced after step


def test_config_validation_temperature_without_softtree():
    """Test that temperature annealing without SoftTree raises validation error."""
    from regnn.macroutils.base import MacroConfig
    from regnn.data import DataFrameReadInConfig, ModeratedRegressionConfig
    from regnn.train import (
        TemperatureAnnealingConfig,
        TrainingHyperParams,
        OptimizerConfig,
    )

    # Create config with temperature annealing but without SoftTree
    with pytest.raises(ValueError, match="Temperature annealing.*use_soft_tree"):
        MacroConfig(
            read_config=DataFrameReadInConfig(read_cols=["a", "b", "c", "d", "e"]),
            regression=ModeratedRegressionConfig(
                focal_predictor="a",
                controlled_cols=["b"],
                moderators=["c", "d"],
                outcome_col="e",
            ),
            model=ReGNNConfig.create(
                num_moderators=2,
                num_controlled=1,
                layer_input_sizes=[8],
                use_soft_tree=False,  # Not using SoftTree
                vae=False,
            ),
            training=TrainingHyperParams(
                optimizer_config=OptimizerConfig(
                    temperature_annealing=TemperatureAnnealingConfig(
                        schedule_type="linear", initial_temp=1.0, final_temp=5.0
                    )
                )
            ),
        )


def test_config_validation_warmup_epochs():
    """Test that warmup_epochs >= total_epochs raises validation error."""
    from regnn.macroutils.base import MacroConfig
    from regnn.data import DataFrameReadInConfig, ModeratedRegressionConfig
    from regnn.train import WarmupCosineConfig, TrainingHyperParams, OptimizerConfig

    with pytest.raises(ValueError, match="Warmup epochs.*must be less than"):
        MacroConfig(
            read_config=DataFrameReadInConfig(read_cols=["a", "b", "c", "d", "e"]),
            regression=ModeratedRegressionConfig(
                focal_predictor="a",
                controlled_cols=["b"],
                moderators=["c", "d"],
                outcome_col="e",
            ),
            model=ReGNNConfig.create(
                num_moderators=2,
                num_controlled=1,
                layer_input_sizes=[8],
                vae=False,
            ),
            training=TrainingHyperParams(
                epochs=5,
                optimizer_config=OptimizerConfig(
                    scheduler=WarmupCosineConfig(
                        warmup_epochs=10,  # More than total epochs
                        T_max=None,
                        eta_min=0.0,
                    )
                ),
            ),
        )
