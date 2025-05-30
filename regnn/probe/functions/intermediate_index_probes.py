# regnn/probe/functions/intermediate_index_probes.py
import torch
import numpy as np
import pandas as pd
import os
from typing import Optional, TypeVar, Any, List

from regnn.model import ReGNN  # Actual model import
from regnn.probe.registry import register_probe
from regnn.probe.dataclass.probe_config import (
    SaveIntermediateIndexProbeScheduleConfig,
    FrequencyType,
)
from regnn.probe.dataclass.results import IntermediateIndexSavedProbeResult
from regnn.data import ReGNNDataset
from regnn.train import TrainingHyperParams

# TrainingHyperParams = TypeVar("TrainingHyperParams")


def compute_index_prediction(
    model: ReGNN, interaction_predictors: torch.Tensor
) -> np.ndarray:
    index_model = model.index_prediction_model
    index_model.to(interaction_predictors.device).eval()
    with torch.no_grad():
        if hasattr(index_model, "vae") and index_model.vae:
            index_prediction_tensor, _ = index_model(interaction_predictors)
        else:
            index_prediction_tensor = index_model(interaction_predictors)
    return index_prediction_tensor.detach().cpu().numpy()


@register_probe("save_intermediate_index")
def save_intermediate_index_probe(
    model: ReGNN,
    schedule_config: SaveIntermediateIndexProbeScheduleConfig,
    epoch: int,
    dataset: Optional[ReGNNDataset] = None,
    data_source_name: Optional[str] = None,
    training_hp: Optional[TrainingHyperParams] = None,
    **kwargs,
) -> Optional[IntermediateIndexSavedProbeResult]:
    current_status = "success"
    status_message = None
    saved_file_path = "N/A"

    device_to_use = "cpu"
    if training_hp:
        device_to_use = training_hp.device

    try:
        all_moderators_tensor = dataset.to_tensor(device=device_to_use).get(
            "moderators"
        )
        if all_moderators_tensor is None:
            raise ValueError("Moderators not found in dataset.to_tensor() result.")

        idx_pred_np = compute_index_prediction(model, all_moderators_tensor)
        df_orig = dataset.df_orig.copy()
        df_orig[schedule_config.index_column_name]
        df_indices = pd.DataFrame(idx_pred_np)

        model_file_base = schedule_config.model_save_name
        file_id_suffix = (
            f"-{schedule_config.file_id}" if schedule_config.file_id else ""
        )

        epoch_part = f"_epoch_{epoch + 1}"
        if schedule_config.frequency_type == FrequencyType.POST_TRAINING:
            epoch_part = "_final"

        indices_filename = f"{model_file_base}{file_id_suffix}{epoch_part}-indices.dta"
        actual_save_dir = schedule_config.save_dir
        saved_file_path = os.path.join(actual_save_dir, indices_filename)

        os.makedirs(actual_save_dir, exist_ok=True)
        df_indices.to_stata(saved_file_path)

    except Exception as e:
        import traceback

        current_status = "failure"
        status_message = (
            f"Error in save_intermediate_index_probe: {e}\n{traceback.format_exc()}"
        )
        print(status_message)
        # saved_file_path might be partially constructed or point to intended dir
        if "actual_save_dir" in locals() and "indices_filename" in locals():
            saved_file_path = (
                f"failed_to_save_at_{os.path.join(actual_save_dir, indices_filename)}"
            )
        else:
            saved_file_path = f"failed_to_save_at_{schedule_config.save_dir}"

    result = IntermediateIndexSavedProbeResult(
        data_source=data_source_name or "unknown",
        file_path=saved_file_path,
        status=current_status,
        message=status_message,
    )
    return result
