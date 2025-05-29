from typing import Optional

# Actual specific types from the project
from ..registry import register_probe
from ..dataclass.probe_config import GetL2LengthProbeScheduleConfig
from ..dataclass.regression import (
    L2NormProbe as L2NormProbeResult,
)
from ..fns import get_l2_length
from regnn.model import ReGNN


@register_probe("l2_length")
def l2_length_probe(
    model: ReGNN,
    schedule_config: GetL2LengthProbeScheduleConfig,
    data_source_name: str,
    **kwargs,
) -> Optional[L2NormProbeResult]:
    """
    Calculates L2 norm of model parameters and returns an L2NormProbeResult.
    The L2NormProbeResult itself will be part of a Snapshot which contains epoch/time info.
    """

    current_status = "success"
    status_message = None
    l2_data_dict = None

    try:
        l2_data_dict = get_l2_length(
            model
        )  # Returns a dict like {"main": float, "index": float}
    except Exception as e:
        import traceback

        current_status = "failure"
        status_message = f"Error in get_l2_length: {e}\n{traceback.format_exc()}"
        print(status_message)  # Or log it
        l2_data_dict = {"main": -1.0, "index": -1.0}  # Default/error values

    # Create the L2NormProbeResult instance
    # data_source is a required field in ProbeData base and its subclasses
    probe_result = L2NormProbeResult(
        data_source=data_source_name,
        main_norm=l2_data_dict.get("main", -1.0),
        index_norm=l2_data_dict.get("index", -1.0),
        status=current_status,
        message=status_message,
    )

    return probe_result
