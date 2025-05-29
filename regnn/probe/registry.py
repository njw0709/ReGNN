from typing import Dict, Callable, Optional, Union, List, TypeVar

# Using TypeVar as a placeholder for ProbeData and its subclasses for now.
# Actual probe functions will use more specific types from .dataclass
from .dataclass import ProbeData

PROBE_REGISTRY: Dict[
    str, Callable[..., Optional[Union[ProbeData, List[ProbeData]]]]
] = {}


def register_probe(name: str) -> Callable:
    """
    Decorator to register a probe function in the PROBE_REGISTRY.

    Args:
        name: The string identifier for the probe. This should match the
              `probe_type` string used in `ProbeScheduleConfig` subclasses
              (e.g., "regression_eval", "save_checkpoint").
    """

    def decorator(
        func: Callable[..., Optional[Union[ProbeData, List[ProbeData]]]],
    ) -> Callable[..., Optional[Union[ProbeData, List[ProbeData]]]]:
        if name in PROBE_REGISTRY:
            print(
                f"Warning: Probe with name '{name}' is being overwritten in the registry."
            )
        PROBE_REGISTRY[name] = func
        return func

    return decorator
