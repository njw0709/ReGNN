from typing import List, Literal, Union, Dict, Type, Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field


class ProbeData(BaseModel):
    """Base class for all probe data classes"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    data_source: Literal["train", "test", "validate", "all"] = Field(
        ...,
        description="Which data source has been used for computing the probe data. Must be one of: train, test, validate, all",
    )


class Snapshot(BaseModel):
    """Snapshot of all probes at certain moment in time."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    time: Union[int, float, Dict[str, float]] = Field(
        default=-1,
        description="When the measurement was made. Can be epoch, iterations, or dictionary of both.",
    )
    measurements: List[ProbeData] = Field(
        default_factory=list, description="List of all probe measurements at t=time."
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
        return [m for m in self.measurements if isinstance(m, probe_type)]

    def add(self, measurement: ProbeData) -> None:
        """
        Add a new measurement to the snapshot.

        Args:
            measurement: The probe data instance to add to measurements

        Example:
            >>> snapshot = Snapshot(time=1)
            >>> probe_data = ObjectiveProbe(loss=0.5, data_source="train")
            >>> snapshot.add(probe_data)
        """
        self.measurements.append(measurement)


class Trajectory(BaseModel):
    """Data structure for trajectory of measurements.

    By default, can contain measurements of multiple probe types.
    Use select() to get a new trajectory with only specific probe type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)

    data: List[Snapshot] = Field(
        default_factory=list, description="List of all measurements made along time."
    )

    @computed_field
    @property
    def times(self) -> List[Union[int, float, Dict[str, float]]]:
        """Returns a list of times from all snapshots in chronological order."""
        return [snapshot.time for snapshot in self.data]

    def at(
        self,
        time: Union[int, float, Dict[str, float]],
        tolerance: float = 1e-10,
    ) -> Union[Snapshot, List[Snapshot], None]:
        """
        Get snapshot(s) at a specific time point.

        Args:
            time: The time point to retrieve snapshots for. Can be:
                - A scalar (int/float) for simple time points
                - A dictionary matching the structure of snapshot times
            tolerance: Floating point tolerance for time comparison (default: 1e-10)

        Returns:
            - Single Snapshot if exact match is found
            - List of Snapshots if multiple matches within tolerance
            - None if no matches found

        Example:
            >>> trajectory = Trajectory()
            >>> # Get snapshot at epoch 1
            >>> snapshot = trajectory.at(1)
            >>> # Get snapshot at specific iteration
            >>> snapshot = trajectory.at({"epoch": 1, "iteration": 0.5})
        """
        matches = []

        for snapshot in self.data:
            if isinstance(time, (int, float)) and isinstance(
                snapshot.time, (int, float)
            ):
                if abs(snapshot.time - time) <= tolerance:
                    matches.append(snapshot)
            elif isinstance(time, dict) and isinstance(snapshot.time, dict):
                # All keys in time must exist in snapshot.time and values must match within tolerance
                if all(
                    k in snapshot.time and abs(snapshot.time[k] - v) <= tolerance
                    for k, v in time.items()
                ):
                    matches.append(snapshot)

        if not matches:
            return None
        elif len(matches) == 1:
            return matches[0]
        return matches

    def extend(self, other: "Trajectory") -> None:
        """
        Extend this trajectory with snapshots from another trajectory.

        Args:
            other: Another Trajectory instance whose snapshots will be added to this one

        Example:
            >>> train_trajectory = Trajectory()
            >>> val_trajectory = Trajectory()
            >>> # After collecting some snapshots...
            >>> train_trajectory.extend(val_trajectory)

        Note:
            The snapshots are added in the order they appear in the other trajectory.
            No sorting or deduplication is performed.
        """
        if not isinstance(other, Trajectory):
            raise TypeError(f"Expected Trajectory, got {type(other)}")
        self.data.extend(other.data)

    def append(
        self,
        snapshots: Union[Snapshot, List[Snapshot], Tuple[Snapshot, List[Snapshot]]],
    ) -> None:
        """
        Append one or more snapshots to the trajectory.

        Args:
            snapshots: Can be one of:
                - A single Snapshot
                - A list of Snapshots
                - A tuple of (epoch_snapshot, batch_snapshots) as returned by process_epoch

        Example:
            >>> trajectory = Trajectory()
            >>> # Append single snapshot
            >>> trajectory.append(snapshot)
            >>> # Append list of snapshots
            >>> trajectory.append([snapshot1, snapshot2])
            >>> # Append epoch and batch snapshots
            >>> trajectory.append((epoch_snapshot, batch_snapshots))
        """
        if isinstance(snapshots, (list, tuple)):
            # Handle tuple from process_epoch
            if (
                isinstance(snapshots, tuple)
                and len(snapshots) == 2
                and isinstance(snapshots[0], Snapshot)
                and isinstance(snapshots[1], list)
            ):
                epoch_snapshot, batch_snapshots = snapshots
                self.data.extend(batch_snapshots)
                self.data.append(epoch_snapshot)
            # Handle list of snapshots
            else:
                self.data.extend(snapshots)
        # Handle single snapshot
        elif isinstance(snapshots, Snapshot):
            self.data.append(snapshots)
        else:
            raise TypeError(
                f"Expected Snapshot, List[Snapshot], or Tuple[Snapshot, List[Snapshot]], got {type(snapshots)}"
            )

    def select(self, probe_type: Type[ProbeData]) -> "Trajectory":
        """
        Creates a new Trajectory instance containing only measurements of the specified probe type.

        Args:
            probe_type: The type of probe to filter for (e.g., ObjectiveProbe, RegressionProbe)

        Returns:
            A new Trajectory instance with only the specified probe type measurements

        Example:
            >>> trajectory = Trajectory(data=[...])  # Contains mixed probe types
            >>> objective_trajectory = trajectory.select(ObjectiveProbe)
        """
        filtered_snapshots = []

        for snapshot in self.data:
            # Filter measurements of the specified type
            filtered_measurements = [
                m for m in snapshot.measurements if isinstance(m, probe_type)
            ]

            # Only include snapshots that have measurements of the requested type
            if filtered_measurements:
                filtered_snapshots.append(
                    Snapshot(time=snapshot.time, measurements=filtered_measurements)
                )

        return Trajectory(data=filtered_snapshots)

    def group_by(self) -> List["Trajectory"]:
        """
        Collects probe data from snapshots into separate trajectories based on probe type.

        Returns:
            List of Trajectory objects, each containing measurements of a specific probe type

        Example:
            >>> trajectory = Trajectory(data=[...])  # Contains mixed probe types
            >>> [obj_traj, reg_traj] = trajectory.group_by()  # Returns separate trajectories
        """
        if not self.data:
            return []

        # Get unique probe types from all measurements
        probe_types = {
            type(measurement)
            for snapshot in self.data
            for measurement in snapshot.measurements
        }

        # Create a new trajectory for each probe type using select()
        trajectories = [self.select(probe_type) for probe_type in probe_types]

        return trajectories
