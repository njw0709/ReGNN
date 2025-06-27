from typing import List, Union, Type, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field, SerializeAsAny
from .probe_config import FrequencyType
from .base import ProbeData
from .nn import ObjectiveProbe
from .regression import (
    OLSModeratedResultsProbe,
    OLSResultsProbe,
    VarianceInflationFactorProbe,
    L2NormProbe,
)
from .results import (
    IntermediateIndexSavedProbeResult,
    EarlyStoppingSignalProbeResult,
    CheckpointSavedProbeResult,
)

ProbeDataType = Union[
    OLSModeratedResultsProbe,
    OLSResultsProbe,
    VarianceInflationFactorProbe,
    L2NormProbe,
    IntermediateIndexSavedProbeResult,
    EarlyStoppingSignalProbeResult,
    CheckpointSavedProbeResult,
    ObjectiveProbe,
]


class Snapshot(BaseModel):
    """Snapshot of all probes at a specific point in the training lifecycle."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        from_attributes=True,
    )

    epoch: int = Field(
        ...,
        description="Epoch number for this snapshot. -1 for pre-training, training_hp.epochs for post-training.",
    )
    iteration_in_epoch: Optional[int] = Field(
        None,
        description="Iteration number within the epoch (None if not applicable, e.g., for EPOCH frequency). Batch index.",
    )
    global_iteration: Optional[int] = Field(
        None,
        description="Global iteration count (None if not applicable, e.g., for PRE_TRAINING or EPOCH with no prior iterations).",
    )
    frequency_context: FrequencyType = Field(
        ...,
        description="The frequency context under which these probes were run (e.g., PRE_TRAINING, EPOCH).",
    )

    measurements: SerializeAsAny[List[ProbeDataType]] = Field(
        default_factory=list,
        description="List of ProbeData results collected during this snapshot.",
    )

    def get(self, probe_type: Type[ProbeData]) -> Optional[ProbeData]:
        """
        Get the first measurement of the specified probe type.
        Args:
            probe_type: The type of probe data to retrieve (e.g., ObjectiveProbe, L2NormProbe)
        Returns:
            The first matching probe data instance, or None if no match is found
        Example:
            >>> snapshot = Snapshot(...)
            >>> objective_data = snapshot.get(ObjectiveProbe)
            >>> if objective_data:
            >>>     print(f"Loss: {objective_data.loss}")
        """
        if not (isinstance(probe_type, type) and issubclass(probe_type, ProbeData)):
            raise TypeError(
                f"probe_type must be a subclass of ProbeData, got {probe_type}"
            )

        return next((m for m in self.measurements if isinstance(m, probe_type)), None)

    def get_all(self, probe_type: Type[ProbeData]) -> List[ProbeData]:
        """
        Get all measurements of the specified probe type.
        Args:
            probe_type: The type of probe data to retrieve (e.g., ObjectiveProbe, L2NormProbe)
        Returns:
            List of all matching probe data instances
        Example:
            >>> snapshot = Snapshot(...)
            >>> all_objectives = snapshot.get_all(ObjectiveProbe)
            >>> for obj in all_objectives:
            >>>     print(f"Loss: {obj.loss}")
        """
        if not (isinstance(probe_type, type) and issubclass(probe_type, ProbeData)):
            raise TypeError(
                f"probe_type must be a subclass of ProbeData, got {probe_type}"
            )

        return [m for m in self.measurements if isinstance(m, probe_type)]

    def add(self, measurement: ProbeData) -> None:
        """
        Add a new measurement to the snapshot.
        Args:
            measurement: The probe data instance to add to measurements
        Example:
            >>> snapshot = Snapshot(epoch=0, frequency_context=FrequencyType.EPOCH)
            >>> probe_data = ObjectiveProbe(loss=0.5, data_source="train")
            >>> snapshot.add(probe_data)
        """
        if not isinstance(measurement, ProbeData):
            raise TypeError(
                f"measurement must be an instance of ProbeData, got {type(measurement)}"
            )
        self.measurements.append(measurement)


class Trajectory(BaseModel):
    """Data structure for trajectory of measurements.

    By default, can contain measurements of multiple probe types.
    Use select() to get a new trajectory with only specific probe type.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        from_attributes=True,
    )

    data: List[Snapshot] = Field(
        default_factory=list, description="List of all snapshots made along time."
    )

    def __getitem__(self, idx: Union[int, slice]) -> Union[Snapshot, "Trajectory"]:
        """
        Enable index-based access to snapshots using square bracket notation.
        Args:
            idx: Integer index or slice object
        Returns:
            - Single Snapshot if integer index is provided
            - New Trajectory containing sliced snapshots if slice is provided
        Example:
            >>> trajectory = Trajectory(data=[...])
            >>> snapshot = trajectory[0]  # Get first snapshot
            >>> sub_trajectory = trajectory[1:5]  # Get snapshots from index 1 to 4
        """
        if isinstance(idx, slice):
            return Trajectory(data=self.data[idx])
        return self.data[idx]

    @computed_field
    @property
    def epochs(self) -> List[int]:
        """Returns a list of epoch numbers from all snapshots."""
        return [snapshot.epoch for snapshot in self.data]

    def get_snapshots_for_epoch(self, epoch: int) -> List[Snapshot]:
        """Get all snapshots recorded for a specific epoch."""
        return [s for s in self.data if s.epoch == epoch]

    def get_snapshots_by_frequency(
        self, frequency_context: FrequencyType
    ) -> List[Snapshot]:
        """Get all snapshots recorded for a specific frequency context."""
        return [s for s in self.data if s.frequency_context == frequency_context]

    def extend(self, other: "Trajectory") -> None:
        """Extend this trajectory with snapshots from another trajectory."""
        if not isinstance(other, Trajectory):
            raise TypeError(f"Expected Trajectory, got {type(other)}")
        self.data.extend(other.data)

    def append(self, snapshot: Snapshot) -> None:
        """Append one snapshot to the trajectory."""
        if not isinstance(snapshot, Snapshot):
            raise TypeError(f"Expected Snapshot, got {type(snapshot)}")
        self.data.append(snapshot)

    def select(self, probe_type: Type[ProbeData]) -> "Trajectory":
        """Creates a new Trajectory instance containing only measurements of the specified probe type from all snapshots."""
        filtered_snapshots = []
        for snapshot in self.data:
            filtered_measurements = snapshot.get_all(probe_type)
            if filtered_measurements:
                new_snapshot = Snapshot(
                    epoch=snapshot.epoch,
                    iteration_in_epoch=snapshot.iteration_in_epoch,
                    global_iteration=snapshot.global_iteration,
                    frequency_context=snapshot.frequency_context,
                    measurements=filtered_measurements,
                )
                filtered_snapshots.append(new_snapshot)
        return Trajectory(data=filtered_snapshots)

    def group_by(self) -> List["Trajectory"]:
        """Collects probe data from snapshots into separate trajectories based on probe type."""
        if not self.data:
            return []
        probe_types = {
            type(measurement)
            for snapshot in self.data
            for measurement in snapshot.measurements
        }
        trajectories = [self.select(probe_type) for probe_type in probe_types]
        return trajectories
