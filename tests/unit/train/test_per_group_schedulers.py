import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from regnn.train import (
    StepLRConfig,
    ExponentialLRConfig,
    CosineAnnealingLRConfig,
    WarmupCosineConfig,
    OptimizerConfig,
    LearningRateConfig,
    WeightDecayConfig,
)
from regnn.macroutils.utils import PerGroupScheduler


class DummyModel(nn.Module):
    """Simple model for testing optimizer and scheduler."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_per_group_scheduler_step_lr():
    """Test PerGroupScheduler with StepLR for both groups."""
    model = DummyModel()
    # Create optimizer with two parameter groups
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 0.1},
            {"params": model.fc2.parameters(), "lr": 0.01},
        ]
    )

    scheduler_configs = {
        "nn": StepLRConfig(step_size=2, gamma=0.5),
        "regression": StepLRConfig(step_size=3, gamma=0.1),
    }
    param_group_indices = {"nn": 0, "regression": 1}

    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices=param_group_indices,
        total_epochs=10,
    )

    # Check initial LRs
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.01

    # Step once - no change yet (step happens at step_size)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.1  # No change yet
    assert optimizer.param_groups[1]["lr"] == 0.01  # No change yet

    # Step to epoch 2 - NN group should decay
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == 0.01  # No change yet

    # Step to epoch 3 - regression group should decay
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.001, abs=1e-6)


def test_per_group_scheduler_mixed_configs():
    """Test PerGroupScheduler with different scheduler types per group."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 1.0},
            {"params": model.fc2.parameters(), "lr": 0.1},
        ]
    )

    scheduler_configs = {
        "nn": ExponentialLRConfig(gamma=0.9),
        "regression": StepLRConfig(step_size=2, gamma=0.5),
    }
    param_group_indices = {"nn": 0, "regression": 1}

    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices=param_group_indices,
        total_epochs=10,
    )

    # Initial LRs
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert optimizer.param_groups[1]["lr"] == 0.1

    # After 1 step: NN decays exponentially, regression unchanged
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.9, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == 0.1

    # After 2 steps: NN continues exponential, regression decays by step
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.81, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05, abs=1e-6)


def test_per_group_scheduler_cosine_annealing():
    """Test PerGroupScheduler with CosineAnnealingLR."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 1.0},
            {"params": model.fc2.parameters(), "lr": 0.5},
        ]
    )

    scheduler_configs = {
        "nn": CosineAnnealingLRConfig(T_max=10, eta_min=0.0),
        "regression": None,  # No scheduler for regression group
    }
    param_group_indices = {"nn": 0, "regression": 1}

    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices=param_group_indices,
        total_epochs=10,
    )

    # Initial LRs
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert optimizer.param_groups[1]["lr"] == 0.5

    # Step 5 times - NN should follow cosine, regression stays constant
    for _ in range(5):
        scheduler.step()

    # At T_max/2, cosine should be near eta_min
    assert optimizer.param_groups[0]["lr"] < 0.1  # Significantly reduced
    assert optimizer.param_groups[1]["lr"] == 0.5  # Unchanged


def test_per_group_scheduler_warmup_cosine():
    """Test PerGroupScheduler with WarmupCosine schedule."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 1.0},
            {"params": model.fc2.parameters(), "lr": 0.1},
        ]
    )

    scheduler_configs = {
        "nn": WarmupCosineConfig(warmup_epochs=5, T_max=10, eta_min=0.0),
        "regression": None,
    }
    param_group_indices = {"nn": 0, "regression": 1}

    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices=param_group_indices,
        total_epochs=15,
    )

    # Initial LR
    assert optimizer.param_groups[0]["lr"] == 1.0

    # During warmup (epoch 0-4), LR should increase from near 0
    scheduler.step()  # epoch 0
    lr_epoch_0 = optimizer.param_groups[0]["lr"]
    assert lr_epoch_0 < 1.0  # Should be warming up

    scheduler.step()  # epoch 1
    lr_epoch_1 = optimizer.param_groups[0]["lr"]
    assert lr_epoch_1 > lr_epoch_0  # Should be increasing

    # Skip to epoch 5 (after warmup)
    for _ in range(3):
        scheduler.step()

    scheduler.step()  # epoch 5
    lr_after_warmup = optimizer.param_groups[0]["lr"]
    assert lr_after_warmup == pytest.approx(1.0, abs=0.1)  # Should be near base LR

    # Continue stepping - should now follow cosine annealing
    for _ in range(5):
        scheduler.step()

    lr_cosine_phase = optimizer.param_groups[0]["lr"]
    assert lr_cosine_phase < lr_after_warmup  # Should be decaying


def test_per_group_scheduler_none_config():
    """Test PerGroupScheduler with None config for one group."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 0.1},
            {"params": model.fc2.parameters(), "lr": 0.01},
        ]
    )

    scheduler_configs = {
        "nn": StepLRConfig(step_size=2, gamma=0.5),
        "regression": None,  # No scheduler
    }
    param_group_indices = {"nn": 0, "regression": 1}

    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices=param_group_indices,
        total_epochs=10,
    )

    # Step multiple times
    for _ in range(5):
        scheduler.step()

    # NN group should decay, regression should stay constant
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == 0.01  # Unchanged


def test_optimizer_config_validation_warning():
    """Test that OptimizerConfig warns when both global and per-group schedulers are set."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = OptimizerConfig(
            lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
            weight_decay=WeightDecayConfig(weight_decay_nn=0.01, weight_decay_regression=0.0),
            scheduler=StepLRConfig(step_size=10, gamma=0.5),
            scheduler_nn=ExponentialLRConfig(gamma=0.95),
        )

        # Check that a warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Per-group schedulers will take precedence" in str(w[0].message)


def test_optimizer_config_per_group_only():
    """Test OptimizerConfig with only per-group schedulers (no warning)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = OptimizerConfig(
            lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
            scheduler_nn=StepLRConfig(step_size=10, gamma=0.5),
            scheduler_regression=ExponentialLRConfig(gamma=0.9),
        )

        # No warning should be issued
        assert len(w) == 0


def test_optimizer_config_global_only():
    """Test OptimizerConfig with only global scheduler (no warning)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = OptimizerConfig(
            lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
            scheduler=StepLRConfig(step_size=10, gamma=0.5),
        )

        # No warning should be issued
        assert len(w) == 0


def test_per_group_scheduler_is_reduce_on_plateau_flag():
    """Test that is_reduce_on_plateau flag is set correctly."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # No ReduceLROnPlateau
    scheduler_configs = {
        "nn": StepLRConfig(step_size=2, gamma=0.5),
        "regression": None,
    }
    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices={"nn": 0, "regression": 1},
        total_epochs=10,
    )
    assert scheduler.is_reduce_on_plateau is False


def test_per_group_scheduler_epoch_tracking():
    """Test that epoch counter is properly tracked."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 1.0},
            {"params": model.fc2.parameters(), "lr": 0.1},
        ]
    )

    scheduler_configs = {
        "nn": ExponentialLRConfig(gamma=0.9),
        "regression": None,
    }
    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices={"nn": 0, "regression": 1},
        total_epochs=10,
    )

    assert scheduler.current_epoch == 0

    scheduler.step()
    assert scheduler.current_epoch == 1

    scheduler.step()
    assert scheduler.current_epoch == 2


def test_per_group_scheduler_base_lr_storage():
    """Test that base LRs are stored correctly."""
    model = DummyModel()
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters(), "lr": 0.123},
            {"params": model.fc2.parameters(), "lr": 0.456},
        ]
    )

    scheduler_configs = {
        "nn": StepLRConfig(step_size=2, gamma=0.5),
        "regression": ExponentialLRConfig(gamma=0.9),
    }
    scheduler = PerGroupScheduler(
        optimizer=optimizer,
        scheduler_configs=scheduler_configs,
        param_group_indices={"nn": 0, "regression": 1},
        total_epochs=10,
    )

    assert scheduler.base_lrs["nn"] == 0.123
    assert scheduler.base_lrs["regression"] == 0.456
