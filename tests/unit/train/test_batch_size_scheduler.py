import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from regnn.train import (
    StepBatchSizeConfig,
    PiecewiseBatchSizeConfig,
    TrainingHyperParams,
)
from regnn.macroutils.utils import BatchSizeScheduler


class TestBatchSizeScheduler:
    """Test suite for BatchSizeScheduler class."""

    def test_step_schedule_basic(self):
        """Test basic step-based batch size scheduling."""
        config = StepBatchSizeConfig(step_size=10, gamma=2, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # Check initial batch size
        assert scheduler.current_batch_size == 32
        assert scheduler.get_batch_size(0) == 32

        # Epoch 0-9: should stay at 32
        for epoch in range(10):
            bs, should_recreate = scheduler.step(epoch)
            if epoch == 0:
                assert bs == 32
                assert should_recreate == False  # No change at epoch 0
            else:
                assert bs == 32
                assert should_recreate == False

        # Epoch 10: should double to 64
        bs, should_recreate = scheduler.step(10)
        assert bs == 64
        assert should_recreate == True
        assert scheduler.current_batch_size == 64

        # Epoch 11-19: should stay at 64
        for epoch in range(11, 20):
            bs, should_recreate = scheduler.step(epoch)
            assert bs == 64
            assert should_recreate == False

        # Epoch 20: should double to 128
        bs, should_recreate = scheduler.step(20)
        assert bs == 128
        assert should_recreate == True

    def test_step_schedule_with_max_batch_size(self):
        """Test step schedule with maximum batch size cap."""
        config = StepBatchSizeConfig(step_size=5, gamma=2, max_batch_size=100)
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # Epoch 0: 32
        assert scheduler.get_batch_size(0) == 32

        # Epoch 5: 64
        assert scheduler.get_batch_size(5) == 64

        # Epoch 10: 128, but capped at 100
        assert scheduler.get_batch_size(10) == 100

        # Epoch 15: 256, but capped at 100
        assert scheduler.get_batch_size(15) == 100

    def test_gamma_equals_three(self):
        """Test with gamma=3 (triple batch size)."""
        config = StepBatchSizeConfig(step_size=5, gamma=3, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=10)

        assert scheduler.get_batch_size(0) == 10
        assert scheduler.get_batch_size(5) == 30
        assert scheduler.get_batch_size(10) == 90
        assert scheduler.get_batch_size(15) == 270

    def test_should_update_method(self):
        """Test should_update method returns correct values."""
        config = StepBatchSizeConfig(step_size=10, gamma=2, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # Epoch 0: no update (first epoch)
        assert scheduler.should_update(0) == False

        # Epoch 1-9: no update
        for epoch in range(1, 10):
            assert scheduler.should_update(epoch) == False

        # Epoch 10: should update
        assert scheduler.should_update(10) == True

        # Epoch 11-19: no update
        for epoch in range(11, 20):
            assert scheduler.should_update(epoch) == False

        # Epoch 20: should update
        assert scheduler.should_update(20) == True

    def test_epoch_tracking(self):
        """Test that epoch counter is properly tracked."""
        config = StepBatchSizeConfig(step_size=5, gamma=2, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        assert scheduler.current_epoch == 0

        scheduler.step(0)
        assert scheduler.current_epoch == 0

        scheduler.step(1)
        assert scheduler.current_epoch == 1

        scheduler.step(10)
        assert scheduler.current_epoch == 10

    def test_recreate_flag_accuracy(self):
        """Test that should_recreate flag is accurate."""
        config = StepBatchSizeConfig(step_size=3, gamma=2, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=16)

        # First call at epoch 0
        bs, should_recreate = scheduler.step(0)
        assert bs == 16
        assert should_recreate == False  # Initial size, no recreation

        # Epoch 1-2: no change
        for epoch in range(1, 3):
            bs, should_recreate = scheduler.step(epoch)
            assert bs == 16
            assert should_recreate == False

        # Epoch 3: change to 32
        bs, should_recreate = scheduler.step(3)
        assert bs == 32
        assert should_recreate == True

        # Epoch 4-5: no change
        for epoch in range(4, 6):
            bs, should_recreate = scheduler.step(epoch)
            assert bs == 32
            assert should_recreate == False

        # Epoch 6: change to 64
        bs, should_recreate = scheduler.step(6)
        assert bs == 64
        assert should_recreate == True

    def test_max_batch_size_prevents_recreation_once_reached(self):
        """Test that once max_batch_size is reached, no more recreation happens."""
        config = StepBatchSizeConfig(step_size=2, gamma=2, max_batch_size=32)
        scheduler = BatchSizeScheduler(config, initial_batch_size=16)

        # Epoch 0: 16
        bs, should_recreate = scheduler.step(0)
        assert bs == 16
        assert should_recreate == False

        # Epoch 2: 32 (hits max)
        bs, should_recreate = scheduler.step(2)
        assert bs == 32
        assert should_recreate == True

        # Epoch 4: still 32 (capped)
        bs, should_recreate = scheduler.step(4)
        assert bs == 32
        assert should_recreate == False  # No change, already at max

        # Epoch 6: still 32 (capped)
        bs, should_recreate = scheduler.step(6)
        assert bs == 32
        assert should_recreate == False


class TestStepBatchSizeConfig:
    """Test suite for StepBatchSizeConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = StepBatchSizeConfig(step_size=10, gamma=2, max_batch_size=1024)
        assert config.step_size == 10
        assert config.gamma == 2
        assert config.max_batch_size == 1024

    def test_step_size_validation(self):
        """Test that step_size must be positive."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StepBatchSizeConfig(step_size=0, gamma=2)

        with pytest.raises(Exception):
            StepBatchSizeConfig(step_size=-5, gamma=2)

    def test_gamma_validation(self):
        """Test that gamma must be > 1."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StepBatchSizeConfig(step_size=10, gamma=1)

        with pytest.raises(Exception):
            StepBatchSizeConfig(step_size=10, gamma=0)

    def test_max_batch_size_optional(self):
        """Test that max_batch_size is optional."""
        config = StepBatchSizeConfig(step_size=10, gamma=2)
        assert config.max_batch_size is None

    def test_max_batch_size_validation(self):
        """Test that max_batch_size must be positive if provided."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StepBatchSizeConfig(step_size=10, gamma=2, max_batch_size=0)

        with pytest.raises(Exception):
            StepBatchSizeConfig(step_size=10, gamma=2, max_batch_size=-100)


class TestTrainingHyperParamsIntegration:
    """Test integration with TrainingHyperParams."""

    def test_training_hyperparams_with_batch_scheduler(self):
        """Test that TrainingHyperParams accepts batch_size_scheduler."""
        batch_scheduler_config = StepBatchSizeConfig(
            step_size=20, gamma=2, max_batch_size=2048
        )

        training_hp = TrainingHyperParams(
            epochs=100,
            batch_size=256,
            batch_size_scheduler=batch_scheduler_config,
        )

        assert training_hp.batch_size == 256
        assert training_hp.batch_size_scheduler is not None
        assert training_hp.batch_size_scheduler.step_size == 20
        assert training_hp.batch_size_scheduler.gamma == 2

    def test_training_hyperparams_without_batch_scheduler(self):
        """Test that batch_size_scheduler is optional."""
        training_hp = TrainingHyperParams(
            epochs=100,
            batch_size=256,
        )

        assert training_hp.batch_size == 256
        assert training_hp.batch_size_scheduler is None


class TestBatchSizeGrowthPatterns:
    """Test various batch size growth patterns."""

    def test_doubling_pattern(self):
        """Test typical doubling pattern: 256 -> 512 -> 1024 -> 2048."""
        config = StepBatchSizeConfig(step_size=20, gamma=2, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=256)

        assert scheduler.get_batch_size(0) == 256
        assert scheduler.get_batch_size(19) == 256
        assert scheduler.get_batch_size(20) == 512
        assert scheduler.get_batch_size(39) == 512
        assert scheduler.get_batch_size(40) == 1024
        assert scheduler.get_batch_size(59) == 1024
        assert scheduler.get_batch_size(60) == 2048

    def test_aggressive_growth(self):
        """Test aggressive growth with gamma=4."""
        config = StepBatchSizeConfig(step_size=10, gamma=4, max_batch_size=None)
        scheduler = BatchSizeScheduler(config, initial_batch_size=64)

        assert scheduler.get_batch_size(0) == 64
        assert scheduler.get_batch_size(10) == 256
        assert scheduler.get_batch_size(20) == 1024
        assert scheduler.get_batch_size(30) == 4096

    def test_conservative_growth_with_cap(self):
        """Test conservative growth with frequent steps and low cap."""
        config = StepBatchSizeConfig(step_size=5, gamma=2, max_batch_size=512)
        scheduler = BatchSizeScheduler(config, initial_batch_size=128)

        assert scheduler.get_batch_size(0) == 128
        assert scheduler.get_batch_size(5) == 256
        assert scheduler.get_batch_size(10) == 512  # Capped
        assert scheduler.get_batch_size(15) == 512  # Still capped
        assert scheduler.get_batch_size(100) == 512  # Always capped


class TestPiecewiseBatchSizeConfig:
    """Test suite for PiecewiseBatchSizeConfig validation."""

    def test_valid_config(self):
        """Test valid piecewise configuration."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        assert config.type == "piecewise"
        assert config.milestones == [(0, 2048), (50, 256)]

    def test_auto_sorts_milestones(self):
        """Test that milestones are auto-sorted by epoch."""
        config = PiecewiseBatchSizeConfig(milestones=[(50, 256), (0, 2048)])
        assert config.milestones == [(0, 2048), (50, 256)]

    def test_single_milestone(self):
        """Test with a single milestone."""
        config = PiecewiseBatchSizeConfig(milestones=[(10, 512)])
        assert config.milestones == [(10, 512)]

    def test_empty_milestones_raises(self):
        """Test that empty milestones list raises validation error."""
        with pytest.raises(Exception):
            PiecewiseBatchSizeConfig(milestones=[])

    def test_negative_epoch_raises(self):
        """Test that negative epoch in milestone raises validation error."""
        with pytest.raises(Exception):
            PiecewiseBatchSizeConfig(milestones=[(-1, 256)])

    def test_zero_batch_size_raises(self):
        """Test that zero batch size in milestone raises validation error."""
        with pytest.raises(Exception):
            PiecewiseBatchSizeConfig(milestones=[(0, 0)])

    def test_duplicate_epochs_raises(self):
        """Test that duplicate epochs in milestones raises validation error."""
        with pytest.raises(Exception):
            PiecewiseBatchSizeConfig(milestones=[(0, 256), (0, 512)])


class TestPiecewiseBatchSizeScheduler:
    """Test suite for BatchSizeScheduler with PiecewiseBatchSizeConfig."""

    def test_warm_start_then_reduce(self):
        """Test warm-start strategy: large batch then small batch."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # Epochs 0-49: batch_size = 2048 (from first milestone)
        assert scheduler.get_batch_size(0) == 2048
        assert scheduler.get_batch_size(25) == 2048
        assert scheduler.get_batch_size(49) == 2048

        # Epochs 50+: batch_size = 256 (from second milestone)
        assert scheduler.get_batch_size(50) == 256
        assert scheduler.get_batch_size(100) == 256

    def test_initial_batch_size_used_before_first_milestone(self):
        """Test that initial_batch_size is used before the first milestone epoch."""
        config = PiecewiseBatchSizeConfig(milestones=[(20, 1024), (50, 256)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=128)

        # Epochs 0-19: uses initial_batch_size since no milestone covers epoch 0
        assert scheduler.get_batch_size(0) == 128
        assert scheduler.get_batch_size(10) == 128
        assert scheduler.get_batch_size(19) == 128

        # Epochs 20-49: batch_size = 1024
        assert scheduler.get_batch_size(20) == 1024
        assert scheduler.get_batch_size(49) == 1024

        # Epochs 50+: batch_size = 256
        assert scheduler.get_batch_size(50) == 256

    def test_three_phase_schedule(self):
        """Test a three-phase schedule: large -> medium -> small."""
        config = PiecewiseBatchSizeConfig(
            milestones=[(0, 2048), (30, 512), (60, 128)]
        )
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        assert scheduler.get_batch_size(0) == 2048
        assert scheduler.get_batch_size(29) == 2048
        assert scheduler.get_batch_size(30) == 512
        assert scheduler.get_batch_size(59) == 512
        assert scheduler.get_batch_size(60) == 128
        assert scheduler.get_batch_size(100) == 128

    def test_step_method_triggers_recreation(self):
        """Test that step() correctly signals DataLoader recreation."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # Epoch 0: initial batch size is 2048 (from milestone), no change needed
        bs, should_recreate = scheduler.step(0)
        assert bs == 2048
        assert should_recreate == False

        # Epochs 1-49: no change
        for epoch in range(1, 50):
            bs, should_recreate = scheduler.step(epoch)
            assert bs == 2048
            assert should_recreate == False

        # Epoch 50: change to 256
        bs, should_recreate = scheduler.step(50)
        assert bs == 256
        assert should_recreate == True

        # Epoch 51: no change
        bs, should_recreate = scheduler.step(51)
        assert bs == 256
        assert should_recreate == False

    def test_should_update_method(self):
        """Test should_update method for piecewise schedule."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        assert scheduler.should_update(0) == False
        assert scheduler.should_update(49) == False
        assert scheduler.should_update(50) == True
        assert scheduler.should_update(51) == False

    def test_current_batch_size_initialized_from_milestone(self):
        """Test that current_batch_size is set from epoch-0 milestone, not initial_batch_size."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=32)

        # current_batch_size should be 2048 (from milestone), not 32
        assert scheduler.current_batch_size == 2048

    def test_current_batch_size_uses_initial_when_no_epoch_zero_milestone(self):
        """Test that current_batch_size uses initial_batch_size when no epoch-0 milestone."""
        config = PiecewiseBatchSizeConfig(milestones=[(20, 1024)])
        scheduler = BatchSizeScheduler(config, initial_batch_size=128)

        assert scheduler.current_batch_size == 128


class TestTrainingHyperParamsWithPiecewise:
    """Test TrainingHyperParams integration with piecewise batch scheduler."""

    def test_piecewise_batch_scheduler(self):
        """Test that TrainingHyperParams accepts PiecewiseBatchSizeConfig."""
        config = PiecewiseBatchSizeConfig(milestones=[(0, 2048), (50, 256)])
        training_hp = TrainingHyperParams(
            epochs=100,
            batch_size=256,
            batch_size_scheduler=config,
        )
        assert training_hp.batch_size_scheduler is not None
        assert training_hp.batch_size_scheduler.type == "piecewise"

    def test_step_batch_scheduler_still_works(self):
        """Test that StepBatchSizeConfig still works (backward compat)."""
        config = StepBatchSizeConfig(step_size=20, gamma=2, max_batch_size=2048)
        training_hp = TrainingHyperParams(
            epochs=100,
            batch_size=256,
            batch_size_scheduler=config,
        )
        assert training_hp.batch_size_scheduler is not None
        assert training_hp.batch_size_scheduler.type == "step"

    def test_regression_gradient_accumulation_default(self):
        """Test that regression_gradient_accumulation_steps defaults to 1."""
        training_hp = TrainingHyperParams(epochs=10, batch_size=32)
        assert training_hp.regression_gradient_accumulation_steps == 1

    def test_regression_gradient_accumulation_custom(self):
        """Test setting custom regression_gradient_accumulation_steps."""
        training_hp = TrainingHyperParams(
            epochs=10,
            batch_size=32,
            regression_gradient_accumulation_steps=8,
        )
        assert training_hp.regression_gradient_accumulation_steps == 8

    def test_regression_gradient_accumulation_validation(self):
        """Test that regression_gradient_accumulation_steps must be >= 1."""
        with pytest.raises(Exception):
            TrainingHyperParams(
                epochs=10,
                batch_size=32,
                regression_gradient_accumulation_steps=0,
            )
