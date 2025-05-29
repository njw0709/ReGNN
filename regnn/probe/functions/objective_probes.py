import torch
import torch.nn as nn
from typing import Optional, Dict, Any, TypeVar, Callable

from regnn.model import ReGNN
from ..registry import register_probe
from ..dataclass.probe_config import (
    GetObjectiveProbeScheduleConfig,
    FrequencyType,
)
from ..dataclass.nn import ObjectiveProbe
from regnn.train import TrainingHyperParams, KLDLossConfig, MSELossConfig
from regnn.macroutils.utils import (
    setup_loss_and_optimizer,
)  # For consistent loss calculation

# Type hint for dataloader
from torch.utils.data import DataLoader


@register_probe("objective")
def objective_probe(
    model: ReGNN,
    schedule_config: GetObjectiveProbeScheduleConfig,
    data_source_name: str,
    dataloader: DataLoader,
    training_hp: TrainingHyperParams,
    epoch: int,
    frequency_type: FrequencyType,
    shared_resource_accessor: Optional[Callable[[str], Any]] = None,
    **kwargs,
) -> Optional[ObjectiveProbe]:

    # Attempt to retrieve pre-computed training objective for epoch-end evaluation
    # Assumes trainer might store data as: (value: float, name: str, breakdown: dict)
    # Default train_data_source_literal to "train"
    train_ds_literal = (
        "train"  # Could be made configurable via schedule_config.probe_params if needed
    )
    if (
        data_source_name == train_ds_literal
        and frequency_type == FrequencyType.EPOCH
        and shared_resource_accessor is not None
    ):
        precomputed_key = f"epoch_objective_{train_ds_literal}_e{epoch}"
        precomputed_data = shared_resource_accessor(precomputed_key)

        if precomputed_data is not None:
            if isinstance(precomputed_data, tuple) and len(precomputed_data) == 3:
                obj_val, obj_name, obj_breakdown = precomputed_data
                if (
                    isinstance(obj_val, (float, int))
                    and isinstance(obj_name, str)
                    and isinstance(obj_breakdown, dict)
                ):
                    return ObjectiveProbe(
                        objective=float(obj_val),
                        objective_name=obj_name,
                        objective_breakdown=obj_breakdown,
                        data_source=data_source_name,
                        status="success",
                        message=f"Used pre-computed objective from shared_resources (key: {precomputed_key}).",
                    )
                else:
                    print(
                        f"Warning: Probe '{schedule_config.probe_type}': Precomputed objective data for key '{precomputed_key}' has unexpected types. Recomputing."
                    )
            else:
                print(
                    f"Warning: Probe '{schedule_config.probe_type}': Precomputed objective data for key '{precomputed_key}' is not a 3-tuple. Recomputing."
                )

    current_status = "success"
    status_message = None
    total_loss_agg = 0.0
    main_loss_agg = 0.0
    reg_loss_agg = 0.0
    num_batches = 0

    try:
        loss_fn_callable, reg_fn_callable, _ = setup_loss_and_optimizer(
            model, training_hp
        )
    except Exception as e:
        import traceback

        return ObjectiveProbe(
            objective=float("nan"),
            objective_name=f"total_loss_on_{data_source_name}",
            objective_breakdown={"error_setup": str(e)},
            data_source=data_source_name,
            status="failure",
            message=f"Failed to setup loss functions: {e}\\n{traceback.format_exc()}",
        )

    model.eval()
    device = training_hp.device if hasattr(training_hp, "device") else "cpu"

    loss_options = training_hp.loss_options
    regnn_model_config = model.config  # This is ReGNNConfig from model internal

    try:
        with torch.no_grad():
            for batch_data in dataloader:
                try:
                    # Assuming batch_data is an object with .to(device) method
                    # and attributes .interaction_predictors and .target
                    # This matches how data is handled in the existing trainer
                    batch_data_on_device = batch_data.to(device)
                    features = batch_data_on_device.interaction_predictors
                    targets = batch_data_on_device.target
                except AttributeError:
                    # Fallback for simpler (features, target) tuple from dataloader
                    if (
                        isinstance(batch_data, (list, tuple))
                        and len(batch_data) == 2
                        and isinstance(batch_data[0], torch.Tensor)
                        and isinstance(batch_data[1], torch.Tensor)
                    ):
                        features, targets = batch_data
                        features = features.to(device)
                        targets = targets.to(device)
                    else:
                        error_msg = "Batch data from dataloader has an unexpected structure. Expected object with .to, .interaction_predictors, .target OR (tensor, tensor) tuple."
                        raise ValueError(error_msg)

                predictions = model(features)

                batch_main_loss_tensor = torch.tensor(0.0, device=device)
                if regnn_model_config.vae and isinstance(loss_options, KLDLossConfig):
                    output_mu, output_log_var = predictions
                    batch_main_loss_tensor = loss_fn_callable(
                        output_mu, output_log_var, targets, output_mu
                    )
                elif isinstance(loss_options, MSELossConfig):
                    batch_main_loss_tensor = loss_fn_callable(predictions, targets)
                else:
                    raise ValueError(
                        f"Unsupported combination of VAE={regnn_model_config.vae} and loss_options={type(loss_options)}"
                    )

                main_loss_agg += batch_main_loss_tensor.item()

                batch_reg_loss_tensor = torch.tensor(0.0, device=device)
                if reg_fn_callable:
                    batch_reg_loss_tensor = reg_fn_callable(model)
                    reg_loss_agg += batch_reg_loss_tensor.item()

                total_loss_agg += (
                    batch_main_loss_tensor.item() + batch_reg_loss_tensor.item()
                )
                num_batches += 1

        if num_batches == 0:
            current_status = "skipped"
            status_message = "Dataloader was empty. No batches processed."
            return ObjectiveProbe(
                objective=float("nan"),
                objective_name=f"total_loss_on_{data_source_name}",
                data_source=data_source_name,
                status=current_status,
                message=status_message,
            )

        avg_total_loss = total_loss_agg / num_batches
        avg_main_loss = main_loss_agg / num_batches
        avg_reg_loss = reg_loss_agg / num_batches

        breakdown = {
            "main_loss": avg_main_loss,
            "regularization_loss": avg_reg_loss,
        }

        result = ObjectiveProbe(
            objective=avg_total_loss,
            objective_name=f"total_loss_on_{data_source_name}",
            objective_breakdown=breakdown,
            data_source=data_source_name,
            status=current_status,
            message=status_message,
        )
        return result

    except Exception as e:
        import traceback

        current_status = "failure"
        status_message = (
            f"Error during objective calculation: {e}\\n{traceback.format_exc()}"
        )
        # print(status_message) # Optionally print for immediate debugging
        return ObjectiveProbe(
            objective=float("nan"),
            objective_name=f"total_loss_on_{data_source_name}",
            objective_breakdown={"error_calculation": str(e)},
            data_source=data_source_name,
            status=current_status,
            message=status_message,
        )
