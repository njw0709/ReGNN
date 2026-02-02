import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from regnn.train import (
    StepLRConfig,
    ExponentialLRConfig,
    CosineAnnealingLRConfig,
    ReduceLROnPlateauConfig,
    WarmupCosineConfig,
)
from regnn.macroutils.utils import create_lr_scheduler


class DummyModel(nn.Module):
    """Simple model for testing optimizer and scheduler."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_create_lr_scheduler_none():
    """Test that None scheduler config returns None."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = create_lr_scheduler(optimizer, None, total_epochs=10)
    assert scheduler is None


def test_create_step_lr_scheduler():
    """Test StepLR scheduler creation."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    config = StepLRConfig(step_size=3, gamma=0.5)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    assert isinstance(scheduler, optim.lr_scheduler.StepLR)

    # Check initial LR
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == 0.1

    # Step through schedule
    for _ in range(3):
        scheduler.step()
    # After 3 steps, LR should be reduced by gamma
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr == pytest.approx(0.05, abs=1e-6)


def test_create_exponential_lr_scheduler():
    """Test ExponentialLR scheduler creation."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = ExponentialLRConfig(gamma=0.9)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    assert isinstance(scheduler, optim.lr_scheduler.ExponentialLR)

    # Check initial LR
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == 1.0

    # Step once
    scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr == pytest.approx(0.9, abs=1e-6)

    # Step again
    scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr == pytest.approx(0.81, abs=1e-6)


def test_create_cosine_annealing_lr_scheduler():
    """Test CosineAnnealingLR scheduler creation."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = CosineAnnealingLRConfig(T_max=10, eta_min=0.0)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

    # Initial LR
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == 1.0

    # Step halfway through cycle
    for _ in range(5):
        scheduler.step()
    mid_lr = optimizer.param_groups[0]["lr"]
    # At T_max/2, cosine should be around the middle (0.5)
    assert 0.4 < mid_lr < 0.6

    # Step to end
    for _ in range(5):
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]
    # At T_max, should be at minimum
    assert final_lr == pytest.approx(0.0, abs=1e-5)


def test_create_cosine_annealing_default_tmax():
    """Test CosineAnnealingLR with default T_max (uses total_epochs)."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = CosineAnnealingLRConfig(T_max=None, eta_min=0.0)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=20)

    assert scheduler is not None
    # T_max should default to total_epochs
    assert scheduler.T_max == 20


def test_create_reduce_on_plateau_scheduler():
    """Test ReduceLROnPlateau scheduler creation."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = ReduceLROnPlateauConfig(
        mode="min", factor=0.5, patience=2, threshold=1e-4
    )
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    # Initial LR
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == 1.0

    # Plateau needs a metric value
    # Simulate non-improving loss (need patience + 1 steps to trigger)
    for _ in range(4):  # patience=2, so need at least 3 steps with no improvement
        scheduler.step(1.0)

    # LR should be reduced after patience
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr == pytest.approx(0.5, abs=1e-6)


def test_create_warmup_cosine_scheduler():
    """Test WarmupCosine scheduler creation."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = WarmupCosineConfig(warmup_epochs=3, T_max=7, eta_min=0.0)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)

    # Initial LR should be very small (warmup start)
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr < 0.01

    # Step through warmup
    for _ in range(3):
        scheduler.step()

    # After warmup, LR should be at full value
    post_warmup_lr = optimizer.param_groups[0]["lr"]
    assert post_warmup_lr == pytest.approx(1.0, abs=1e-2)


def test_create_warmup_cosine_default_tmax():
    """Test WarmupCosine with default T_max."""
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    config = WarmupCosineConfig(warmup_epochs=3, T_max=None, eta_min=0.0)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    assert scheduler is not None
    # T_max should default to total_epochs - warmup_epochs = 7


def test_step_lr_config_validation():
    """Test StepLR config field validation."""
    # Valid config
    config = StepLRConfig(step_size=5, gamma=0.5)
    assert config.step_size == 5
    assert config.gamma == 0.5

    # Invalid step_size (must be > 0)
    with pytest.raises(ValueError):
        StepLRConfig(step_size=0, gamma=0.5)

    # Invalid gamma (must be > 0 and <= 1)
    with pytest.raises(ValueError):
        StepLRConfig(step_size=5, gamma=0.0)

    with pytest.raises(ValueError):
        StepLRConfig(step_size=5, gamma=1.5)


def test_exponential_lr_config_validation():
    """Test ExponentialLR config field validation."""
    # Valid config
    config = ExponentialLRConfig(gamma=0.95)
    assert config.gamma == 0.95

    # Invalid gamma
    with pytest.raises(ValueError):
        ExponentialLRConfig(gamma=0.0)

    with pytest.raises(ValueError):
        ExponentialLRConfig(gamma=1.5)


def test_cosine_annealing_config_validation():
    """Test CosineAnnealingLR config field validation."""
    # Valid config
    config = CosineAnnealingLRConfig(T_max=10, eta_min=0.0)
    assert config.T_max == 10
    assert config.eta_min == 0.0

    # Invalid T_max
    with pytest.raises(ValueError):
        CosineAnnealingLRConfig(T_max=0, eta_min=0.0)

    # Invalid eta_min (must be >= 0)
    with pytest.raises(ValueError):
        CosineAnnealingLRConfig(T_max=10, eta_min=-0.1)


def test_reduce_on_plateau_config_validation():
    """Test ReduceLROnPlateau config field validation."""
    # Valid config
    config = ReduceLROnPlateauConfig(
        mode="min", factor=0.5, patience=5, threshold=1e-4
    )
    assert config.mode == "min"
    assert config.factor == 0.5
    assert config.patience == 5

    # Invalid factor (must be 0 < factor < 1)
    with pytest.raises(ValueError):
        ReduceLROnPlateauConfig(mode="min", factor=0.0, patience=5, threshold=1e-4)

    with pytest.raises(ValueError):
        ReduceLROnPlateauConfig(mode="min", factor=1.5, patience=5, threshold=1e-4)

    # Invalid patience (must be >= 0)
    with pytest.raises(ValueError):
        ReduceLROnPlateauConfig(mode="min", factor=0.5, patience=-1, threshold=1e-4)


def test_warmup_cosine_config_validation():
    """Test WarmupCosine config field validation."""
    # Valid config
    config = WarmupCosineConfig(warmup_epochs=5, T_max=10, eta_min=0.0)
    assert config.warmup_epochs == 5
    assert config.T_max == 10

    # Invalid warmup_epochs (must be > 0)
    with pytest.raises(ValueError):
        WarmupCosineConfig(warmup_epochs=0, T_max=10, eta_min=0.0)


def test_scheduler_with_multiple_param_groups():
    """Test that schedulers work with multiple parameter groups."""
    model = DummyModel()
    # Create optimizer with two parameter groups
    optimizer = optim.Adam(
        [
            {"params": [model.fc.weight], "lr": 0.1},
            {"params": [model.fc.bias], "lr": 0.01},
        ]
    )

    config = StepLRConfig(step_size=2, gamma=0.5)
    scheduler = create_lr_scheduler(optimizer, config, total_epochs=10)

    # Check initial LRs
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.01

    # Step scheduler
    for _ in range(2):
        scheduler.step()

    # Both groups should be scaled by gamma
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05, abs=1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.005, abs=1e-6)
