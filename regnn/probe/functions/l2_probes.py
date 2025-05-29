from typing import Optional, Dict

# Actual specific types from the project
from ..registry import register_probe
from ..dataclass.probe_config import GetL2LengthProbeScheduleConfig
from ..dataclass.regression import (
    L2NormProbe as L2NormProbeResult,
)
from regnn.model import ReGNN
import torch


def get_l2_length(model: ReGNN) -> Dict[str, float]:
    """Calculate L2 norms for model parameters"""
    l2_lengths = {}
    main_parameters = [model.focal_predictor_main_weight, model.predicted_index_weight]
    if model.has_linear_terms:
        main_parameters += [p for p in model.linear_weights.parameters()][:-1]
    main_parameters = torch.cat(main_parameters, dim=1)
    main_param_l2 = main_parameters.norm(2).item()

    index_norm = model.predicted_index_weight.norm(2).item()
    l2_lengths["main"] = main_param_l2
    l2_lengths["index"] = index_norm
    return l2_lengths


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
        l2_data_dict = get_l2_length(model)
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
