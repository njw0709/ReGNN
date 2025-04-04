from typing import List, Literal, Union, Dict, Type
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
        -1,
        "When the measurement was made. Can be epoch, iterations, or dictionary of both.",
    )
    measurements: List[ProbeData] = Field(
        [], "List of all probe measurements at t=time."
    )


class Trajectory(BaseModel):
    """Data structure for trajectory of measurements.

    By default, can contain measurements of multiple probe types.
    Use select() to get a new trajectory with only specific probe type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)

    data: List[Snapshot] = Field([], "List of all measurements made along time.")

    @computed_field
    @property
    def times(self) -> List[Union[int, float, Dict[str, float]]]:
        """Returns a list of times from all snapshots in chronological order."""
        return [snapshot.time for snapshot in self.data]

    def select(self, probe_type: Type[ProbeData]) -> "Trajectory":
        """
        Creates a new Trajectory instance containing only measurements of the specified probe type.

        Args:
            probe_type: The type of probe to filter for (e.g., ObjectiveProbe, RegressionProbe)

        Returns:
            A new Trajectory instance with only the specified probe type measurements

        Example:
            >>> trajectory = Trajectory(data=[...])  # Contains mixed probe types
            >>> objective_trajectory = trajectory.filter_by_probe_type(ObjectiveProbe)
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
