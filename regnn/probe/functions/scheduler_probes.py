"""Probe functions for monitoring learning rate schedulers and temperature annealing."""

import torch
from typing import Optional, Any, Callable
import torch.optim as optim

from regnn.model import ReGNN
from ..registry import register_probe
from ..dataclass.probe_config import FrequencyType
from ..dataclass.nn import SchedulerProbe
from regnn.train import TrainingHyperParams


@register_probe("scheduler_monitor")
def scheduler_monitor_probe(
    model: ReGNN,
    schedule_config: Any,  # Could be a specific SchedulerProbeScheduleConfig if needed
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    temperature_annealer: Optional[Any],
    training_hp: TrainingHyperParams,
    epoch: int,
    frequency_type: FrequencyType,
    **kwargs,
) -> Optional[SchedulerProbe]:
    """Monitor current learning rates and temperature.

    Args:
        model: ReGNN model instance
        schedule_config: Probe schedule configuration
        optimizer: Optimizer being used for training
        scheduler: LR scheduler instance (if any)
        temperature_annealer: Temperature annealer instance (if any)
        training_hp: Training hyperparameters
        epoch: Current epoch
        frequency_type: Frequency at which probe is called
        **kwargs: Additional arguments

    Returns:
        SchedulerProbe with current LRs and temperature, or None
    """
    # Get learning rates from optimizer parameter groups
    # Typically: group 0 = NN params, group 1 = regression params
    lr_nn = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else 0.0
    lr_regression = (
        optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_nn
    )

    # Get current temperature if applicable
    temperature = None
    if temperature_annealer is not None:
        try:
            temperature = temperature_annealer.get_temperature(epoch)
        except Exception:
            # If there's an error getting temperature, just leave it as None
            pass

    # Get scheduler type if available
    scheduler_type = None
    if scheduler is not None:
        scheduler_type = type(scheduler).__name__

    return SchedulerProbe(
        data_source="TRAIN",  # Scheduler info is training-specific
        lr_nn=lr_nn,
        lr_regression=lr_regression,
        temperature=temperature,
        scheduler_type=scheduler_type,
    )
