from typing import List, Dict, Callable, Union, Type, Optional
from regnn.probe.dataclass import (
    ProbeData,
    Snapshot,
    ObjectiveProbe,
    L2NormProbe,
)

# Standard probe collections
DEFAULT_PROBES = [ObjectiveProbe]
MODEL_PROBES = [ObjectiveProbe, L2NormProbe]


def collect_probes(
    probes: Optional[List[Union[Type[ProbeData], Callable]]] = None,
) -> List[Union[Type[ProbeData], Callable]]:
    """
    Helper function to collect requested probes.
    If None is provided, returns DEFAULT_PROBES.

    Args:
        probes: List of probe types or probe functions, or None

    Returns:
        List of probe types or probe functions
    """
    if probes is None:
        return DEFAULT_PROBES.copy()
    return probes


def convert_probe_dict_to_snapshot(
    probe_data: Dict,
    time_value: Union[int, float, Dict] = -1,
    data_source: str = "train",
) -> Snapshot:
    """
    Convert a dictionary of probe results to a Snapshot object.

    Args:
        probe_data: Dictionary of probe results
        time_value: Time value for the snapshot
        data_source: Data source for the probes

    Returns:
        Snapshot object containing the probe data
    """
    snapshot = Snapshot(time=time_value, measurements=[])

    # Handle loss value if present
    if "loss" in probe_data:
        snapshot.measurements.append(
            ObjectiveProbe(data_source=data_source, loss=probe_data["loss"])
        )

    # Handle L2 length data if present
    if "l2_lengths" in probe_data and probe_data["l2_lengths"] is not None:
        l2_probe = L2NormProbe.from_dict(
            probe_data["l2_lengths"], data_source=data_source
        )
        snapshot.measurements.append(l2_probe)

    return snapshot
