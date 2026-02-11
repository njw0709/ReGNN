"""
Test early stopping functionality to ensure it stops at the right time without training extra epochs.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from regnn.probe.dataclass.probe_config import (
    PValEarlyStoppingProbeScheduleConfig,
    DataSource,
    FrequencyType,
)
from regnn.probe.dataclass.results import EarlyStoppingSignalProbeResult
from regnn.probe.dataclass.regression import OLSModeratedResultsProbe
from regnn.probe.dataclass.trajectory import Trajectory, Snapshot
from regnn.probe.functions.stopping_probes import pval_early_stopping_probe


class TestEarlyStoppingCurrentEpochFix:
    """Test that early stopping includes current epoch's data to avoid training one extra epoch."""

    def test_stops_immediately_when_criterion_met_with_current_epoch(self):
        """
        Test that early stopping triggers after epoch 9 when epochs 7, 8, 9 meet criterion,
        preventing unnecessary training of epoch 10.
        """
        # Setup: n_sequential_evals_to_pass=3, patience=5
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST],
            data_sources_to_monitor=[DataSource.TEST],
            n_sequential_evals_to_pass=3,
            patience=5,
            criterion=0.05,
        )

        # Create trajectory with epochs 0-8 (epoch 9 not yet added)
        trajectory = Trajectory()
        
        # Add epochs 0-8 to trajectory
        # Epochs 6, 7, 8 have p-values < 0.05 (meet criterion)
        for epoch in range(9):
            pval = 0.03 if epoch >= 6 else 0.10  # Epochs 6, 7, 8 pass
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=pval,
                    )
                ],
            )
            trajectory.append(snapshot)

        # Create current measurements for epoch 9 (also passes criterion)
        current_measurements = [
            OLSModeratedResultsProbe(
                data_source=DataSource.TEST,
                status="success",
                message="Test",
                interaction_pval=0.03,  # Epoch 9 also passes
            )
        ]

        # Mock shared resource accessor
        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return current_measurements
            return None

        # Run early stopping probe at epoch 9
        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=9,
            shared_resource_accessor=mock_accessor,
        )

        # Should signal to stop because epochs 7, 8, 9 all meet criterion
        assert result is not None
        assert isinstance(result, EarlyStoppingSignalProbeResult)
        assert result.should_stop is True
        # Since patience=5, joint epochs are 5,6,7,8,9 and we check last 3: 7,8,9
        assert "7" in result.reason and "8" in result.reason and "9" in result.reason

    def test_does_not_stop_without_current_epoch_data(self):
        """
        Test that without current epoch's data, the probe wouldn't trigger
        (demonstrating the old bug).
        """
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST],
            data_sources_to_monitor=[DataSource.TEST],
            n_sequential_evals_to_pass=3,
            patience=5,
            criterion=0.05,
        )

        # Create trajectory with epochs 0-8
        trajectory = Trajectory()
        for epoch in range(9):
            # Only epoch 8 passes (epochs 7, 9 would pass but 9 not in trajectory)
            pval = 0.03 if epoch == 8 else 0.10
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=pval,
                    )
                ],
            )
            trajectory.append(snapshot)

        # No current measurements (old behavior)
        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return None  # Or empty list
            return None

        # Run at epoch 9
        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=9,
            shared_resource_accessor=mock_accessor,
        )

        # Should NOT stop because only epoch 8 passes (need 3 consecutive)
        assert result is not None
        assert result.should_stop is False

    def test_respects_patience_period(self):
        """Test that early stopping respects the patience period."""
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST],
            data_sources_to_monitor=[DataSource.TEST],
            n_sequential_evals_to_pass=3,
            patience=10,  # High patience
            criterion=0.05,
        )

        trajectory = Trajectory()
        # Create trajectory with only 5 epochs, all passing
        for epoch in range(5):
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=0.01,  # All pass
                    )
                ],
            )
            trajectory.append(snapshot)

        # Current measurements for epoch 5
        current_measurements = [
            OLSModeratedResultsProbe(
                data_source=DataSource.TEST,
                status="success",
                message="Test",
                interaction_pval=0.01,
            )
        ]

        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return current_measurements
            return None

        # Run at epoch 5 (< patience of 10)
        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=5,
            shared_resource_accessor=mock_accessor,
        )

        # Should not stop because we're still in patience period
        assert result is not None
        assert result.should_stop is False
        assert "patience" in result.message.lower()

    def test_handles_multiple_data_sources(self):
        """Test early stopping with multiple monitored data sources."""
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST, DataSource.TRAIN],
            data_sources_to_monitor=[DataSource.TEST, DataSource.TRAIN],
            n_sequential_evals_to_pass=2,
            patience=3,
            criterion=0.05,
        )

        trajectory = Trajectory()
        # Epochs 3-4: both test and train pass
        for epoch in range(5):
            test_pval = 0.03 if epoch >= 3 else 0.10
            train_pval = 0.04 if epoch >= 3 else 0.10
            
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=test_pval,
                    ),
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TRAIN,
                        status="success",
                        message="Train",
                        interaction_pval=train_pval,
                    ),
                ],
            )
            trajectory.append(snapshot)

        # Current measurements for epoch 5 (both pass)
        current_measurements = [
            OLSModeratedResultsProbe(
                data_source=DataSource.TEST,
                status="success",
                message="Test",
                interaction_pval=0.02,
            ),
            OLSModeratedResultsProbe(
                data_source=DataSource.TRAIN,
                status="success",
                message="Train",
                interaction_pval=0.03,
            ),
        ]

        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return current_measurements
            return None

        # Run at epoch 5
        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=5,
            shared_resource_accessor=mock_accessor,
        )

        # Should stop because last 2 epochs (4, 5) both have all sources passing
        assert result is not None
        assert result.should_stop is True

    def test_does_not_stop_when_one_source_fails(self):
        """Test that all monitored sources must pass for early stopping."""
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST, DataSource.TRAIN],
            data_sources_to_monitor=[DataSource.TEST, DataSource.TRAIN],
            n_sequential_evals_to_pass=2,
            patience=3,
            criterion=0.05,
        )

        trajectory = Trajectory()
        for epoch in range(5):
            # Test passes but train fails
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=0.01,  # Passes
                    ),
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TRAIN,
                        status="success",
                        message="Train",
                        interaction_pval=0.10,  # Fails
                    ),
                ],
            )
            trajectory.append(snapshot)

        current_measurements = [
            OLSModeratedResultsProbe(
                data_source=DataSource.TEST,
                status="success",
                message="Test",
                interaction_pval=0.01,
            ),
            OLSModeratedResultsProbe(
                data_source=DataSource.TRAIN,
                status="success",
                message="Train",
                interaction_pval=0.10,  # Still fails
            ),
        ]

        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return current_measurements
            return None

        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=5,
            shared_resource_accessor=mock_accessor,
        )

        # Should not stop because train source fails
        assert result is not None
        assert result.should_stop is False

    def test_insufficient_historical_data(self):
        """Test behavior when there's not enough historical data yet."""
        schedule_config = PValEarlyStoppingProbeScheduleConfig(
            probe_type="pval_early_stopping",
            frequency_type=FrequencyType.EPOCH,
            frequency_value=1,
            data_sources=[DataSource.TEST],
            data_sources_to_monitor=[DataSource.TEST],
            n_sequential_evals_to_pass=5,  # Need 5 consecutive
            patience=2,
            criterion=0.05,
        )

        # Only 3 epochs in trajectory
        trajectory = Trajectory()
        for epoch in range(3):
            snapshot = Snapshot(
                epoch=epoch,
                iteration_in_epoch=None,
                global_iteration=epoch,
                frequency_context=FrequencyType.EPOCH,
                measurements=[
                    OLSModeratedResultsProbe(
                        data_source=DataSource.TEST,
                        status="success",
                        message="Test",
                        interaction_pval=0.01,
                    )
                ],
            )
            trajectory.append(snapshot)

        current_measurements = [
            OLSModeratedResultsProbe(
                data_source=DataSource.TEST,
                status="success",
                message="Test",
                interaction_pval=0.01,
            )
        ]

        def mock_accessor(key):
            if key == "probe_trajectory":
                return trajectory
            elif key == "current_measurements":
                return current_measurements
            return None

        result = pval_early_stopping_probe(
            schedule_config=schedule_config,
            epoch=3,
            shared_resource_accessor=mock_accessor,
        )

        # Should not stop - only have 4 epochs (0,1,2,3) but need 5
        assert result is not None
        assert result.should_stop is False
        assert "Not enough" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
