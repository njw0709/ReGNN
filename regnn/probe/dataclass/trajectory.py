from typing import List, Literal, Union, Dict, Type, Tuple, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, computed_field, SerializeAsAny
from .base import ProbeData
from .nn import ObjectiveProbe
from .regression import (
    OLSResultsProbe,
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
    L2NormProbe,
)


class Snapshot(BaseModel):
    """Snapshot of all probes at certain moment in time."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
    )

    time: Union[int, float, Dict[str, float]] = Field(
        default=-1,
        description="When the measurement was made. Can be epoch, iterations, or dictionary of both.",
    )
    measurements: List[
        SerializeAsAny[
            Union[
                ObjectiveProbe,
                OLSModeratedResultsProbe,
                OLSResultsProbe,
                VarianceInflationFactorProbe,
                L2NormProbe,
            ]
        ]
    ] = Field(default_factory=list)

    # @classmethod
    # def model_validate_json(
    #     cls, json_data: Union[str, bytes, bytearray, Dict], **kwargs
    # ):
    #     """Override to handle polymorphic deserialization of ProbeData in measurements."""
    #     data_dict: Dict[str, Any]
    #     if isinstance(json_data, dict):
    #         data_dict = json_data
    #     elif isinstance(json_data, (str, bytes, bytearray)):
    #         import json

    #         data_dict = json.loads(
    #             json_data.decode()
    #             if isinstance(json_data, (bytes, bytearray))
    #             else json_data
    #         )
    #     else:
    #         raise TypeError(
    #             f"Expected str, bytes, bytearray or dict for json_data, got {type(json_data)}"
    #         )

    #     if "measurements" in data_dict and isinstance(data_dict["measurements"], list):
    #         processed_measurements = []
    #         for measurement_data in data_dict["measurements"]:
    #             # measurement_data should be a dict here after json.loads if json_data was a string
    #             # or directly if json_data was a pre-parsed dict.
    #             # Pass it to ProbeData.model_validate_json which expects a dict or json string.
    #             processed_measurements.append(
    #                 ProbeData.model_validate_json(measurement_data, **kwargs)
    #             )
    #         data_dict["measurements"] = processed_measurements

    #     return cls.model_validate(data_dict, **kwargs)

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
    )

    data: List[Snapshot] = Field(
        default_factory=list, description="List of all measurements made along time."
    )

    # @classmethod
    # def model_validate_json(
    #     cls, json_data: Union[str, bytes, bytearray, Dict], **kwargs
    # ):
    #     """Override to handle polymorphic deserialization of Snapshots in data."""
    #     data_dict: Dict[str, Any]
    #     if isinstance(json_data, dict):
    #         data_dict = json_data
    #     elif isinstance(json_data, (str, bytes, bytearray)):
    #         import json

    #         data_dict = json.loads(
    #             json_data.decode()
    #             if isinstance(json_data, (bytes, bytearray))
    #             else json_data
    #         )
    #     else:
    #         raise TypeError(
    #             f"Expected str, bytes, bytearray or dict for json_data, got {type(json_data)}"
    #         )

    #     if "data" in data_dict and isinstance(data_dict["data"], list):
    #         processed_data = []
    #         for snapshot_data in data_dict["data"]:
    #             # Pass to Snapshot.model_validate_json
    #             processed_data.append(
    #                 Snapshot.model_validate_json(snapshot_data, **kwargs)
    #             )
    #         data_dict["data"] = processed_data

    #     return cls.model_validate(data_dict, **kwargs)

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
    def times(self) -> List[Union[int, float, Dict[str, float]]]:
        """Returns a list of times from all snapshots in chronological order."""
        return [snapshot.time for snapshot in self.data]

    def at(
        self,
        time: Union[int, float, Dict[str, float]],
        tolerance: float = 1e-10,
    ) -> Union[Snapshot, List[Snapshot], None]:
        """Get snapshot(s) at a specific time point."""
        matches = []
        for snapshot in self.data:
            if isinstance(time, (int, float)) and isinstance(
                snapshot.time, (int, float)
            ):
                if abs(snapshot.time - time) <= tolerance:
                    matches.append(snapshot)
            elif isinstance(time, dict) and isinstance(snapshot.time, dict):
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
        """Extend this trajectory with snapshots from another trajectory."""
        if not isinstance(other, Trajectory):
            raise TypeError(f"Expected Trajectory, got {type(other)}")
        self.data.extend(other.data)

    def append(
        self,
        snapshots: Union[Snapshot, List[Snapshot], Tuple[Snapshot, List[Snapshot]]],
    ) -> None:
        """Append one or more snapshots to the trajectory."""
        if isinstance(snapshots, (list, tuple)):
            if (
                isinstance(snapshots, tuple)
                and len(snapshots) == 2
                and isinstance(snapshots[0], Snapshot)
                and isinstance(snapshots[1], list)
            ):
                epoch_snapshot, batch_snapshots = snapshots
                self.data.extend(batch_snapshots)
                self.data.append(epoch_snapshot)
            else:
                self.data.extend(snapshots)
        elif isinstance(snapshots, Snapshot):
            self.data.append(snapshots)
        else:
            raise TypeError(
                f"Expected Snapshot, List[Snapshot], or Tuple[Snapshot, List[Snapshot]], got {type(snapshots)}"
            )

    def select(self, probe_type: Type[ProbeData]) -> "Trajectory":
        """Creates a new Trajectory instance containing only measurements of the specified probe type."""
        filtered_snapshots = []
        for snapshot in self.data:
            filtered_measurements = [
                m for m in snapshot.measurements if isinstance(m, probe_type)
            ]
            if filtered_measurements:
                filtered_snapshots.append(
                    Snapshot(time=snapshot.time, measurements=filtered_measurements)
                )
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
