import pytest
from typing import Dict, List, Type
from pydantic import ValidationError

from regnn.probe.base import ProbeData, Snapshot, Trajectory


# Define some simple test probe classes
class ProbeType1(ProbeData):
    value: float
    name: str = "probe1"


class ProbeType2(ProbeData):
    score: int
    name: str = "probe2"


@pytest.fixture
def mixed_trajectory():
    """
    Create a trajectory with mixed probe types for testing.
    Timeline: t=1, t=2, t=3, t={"epoch": 4, "iter": 400}
    """
    return Trajectory(
        data=[
            Snapshot(
                time=1,
                measurements=[
                    ProbeType1(data_source="train", value=0.5),
                    ProbeType2(data_source="train", score=10),
                ],
            ),
            Snapshot(
                time=2,
                measurements=[
                    ProbeType1(data_source="test", value=0.4),
                    ProbeType2(data_source="test", score=15),
                ],
            ),
            Snapshot(
                time=3,
                measurements=[
                    ProbeType1(data_source="validate", value=0.6),
                    ProbeType2(data_source="validate", score=20),
                ],
            ),
            Snapshot(
                time={"epoch": 4, "iter": 400},
                measurements=[
                    ProbeType1(data_source="all", value=0.3),
                    ProbeType2(data_source="all", score=25),
                ],
            ),
        ]
    )


class TestProbeData:
    """Tests for ProbeData base class"""

    def test_probe_data_validation(self):
        """Test validation of data_source in ProbeData"""
        # Valid data sources
        ProbeType1(data_source="train", value=0.5)  # Should not raise
        ProbeType1(data_source="test", value=0.5)  # Should not raise
        ProbeType1(data_source="validate", value=0.5)  # Should not raise
        ProbeType1(data_source="all", value=0.5)  # Should not raise

        # Invalid data source
        with pytest.raises(ValidationError):
            ProbeType1(data_source="invalid", value=0.5)


class TestSnapshot:
    """Tests for Snapshot class"""

    def test_snapshot_creation(self):
        """Test creation of Snapshot with different time formats"""
        # Integer time
        snapshot = Snapshot(
            time=1, measurements=[ProbeType1(data_source="train", value=0.5)]
        )
        assert snapshot.time == 1

        # Float time
        snapshot = Snapshot(
            time=1.5, measurements=[ProbeType1(data_source="train", value=0.5)]
        )
        assert snapshot.time == 1.5

        # Dictionary time
        time_dict = {"epoch": 1, "iteration": 100}
        snapshot = Snapshot(
            time=time_dict, measurements=[ProbeType1(data_source="train", value=0.5)]
        )
        assert snapshot.time == time_dict

    def test_snapshot_measurements(self):
        """Test accessing measurements in a snapshot"""
        probe1 = ProbeType1(data_source="train", value=0.5)
        probe2 = ProbeType2(data_source="train", score=10)

        snapshot = Snapshot(time=1, measurements=[probe1, probe2])

        assert len(snapshot.measurements) == 2
        assert isinstance(snapshot.measurements[0], ProbeType1)
        assert isinstance(snapshot.measurements[1], ProbeType2)
        assert snapshot.measurements[0].value == 0.5
        assert snapshot.measurements[1].score == 10


class TestTrajectory:
    """Tests for Trajectory class"""

    def test_times_property(self, mixed_trajectory):
        """Test the times computed property"""
        times = mixed_trajectory.times

        assert len(times) == 4
        assert times[0] == 1
        assert times[1] == 2
        assert times[2] == 3
        assert times[3] == {"epoch": 4, "iter": 400}

    def test_select_method(self, mixed_trajectory):
        """Test filtering trajectory by probe type"""
        # Select TestProbe1 measurements
        probe1_trajectory = mixed_trajectory.select(ProbeType1)

        assert len(probe1_trajectory.data) == 4

        # Check all snapshots have only TestProbe1 measurements
        for snapshot in probe1_trajectory.data:
            assert len(snapshot.measurements) == 1
            assert isinstance(snapshot.measurements[0], ProbeType1)

        # Check values are correct
        assert probe1_trajectory.data[0].measurements[0].value == 0.5
        assert probe1_trajectory.data[1].measurements[0].value == 0.4
        assert probe1_trajectory.data[2].measurements[0].value == 0.6
        assert probe1_trajectory.data[3].measurements[0].value == 0.3

        # Select TestProbe2 measurements
        probe2_trajectory = mixed_trajectory.select(ProbeType2)

        # Check all snapshots have only TestProbe2 measurements
        for snapshot in probe2_trajectory.data:
            assert len(snapshot.measurements) == 1
            assert isinstance(snapshot.measurements[0], ProbeType2)

        # Check values are correct
        assert probe2_trajectory.data[0].measurements[0].score == 10
        assert probe2_trajectory.data[1].measurements[0].score == 15
        assert probe2_trajectory.data[2].measurements[0].score == 20
        assert probe2_trajectory.data[3].measurements[0].score == 25

    def test_select_nonexistent_type(self, mixed_trajectory):
        """Test selecting a probe type that doesn't exist in the trajectory"""

        # Define a new probe type that isn't in the trajectory
        class UnusedProbe(ProbeData):
            data: str

        # Select measurements of a type that doesn't exist
        empty_trajectory = mixed_trajectory.select(UnusedProbe)

        # Should return an empty trajectory
        assert len(empty_trajectory.data) == 0

    def test_group_by_method(self, mixed_trajectory):
        """Test grouping trajectory by probe type"""
        # Group by probe type
        trajectories = mixed_trajectory.group_by()

        # Should return 2 trajectories (one for each probe type)
        assert len(trajectories) == 2

        # Identify which trajectory is which
        if isinstance(trajectories[0].data[0].measurements[0], ProbeType1):
            probe1_trajectory, probe2_trajectory = trajectories
        else:
            probe2_trajectory, probe1_trajectory = trajectories

        # Check TestProbe1 trajectory
        assert len(probe1_trajectory.data) == 4
        for snapshot in probe1_trajectory.data:
            assert len(snapshot.measurements) == 1
            assert isinstance(snapshot.measurements[0], ProbeType1)

        # Check TestProbe2 trajectory
        assert len(probe2_trajectory.data) == 4
        for snapshot in probe2_trajectory.data:
            assert len(snapshot.measurements) == 1
            assert isinstance(snapshot.measurements[0], ProbeType2)

    def test_empty_trajectory(self):
        """Test methods on an empty trajectory"""
        empty_trajectory = Trajectory()

        # Times should be empty
        assert empty_trajectory.times == []

        # Select should return empty
        filtered = empty_trajectory.select(ProbeType1)
        assert len(filtered.data) == 0

        # Group_by should return empty list
        groups = empty_trajectory.group_by()
        assert groups == []
