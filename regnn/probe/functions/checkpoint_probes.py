# regnn/probe/functions/checkpoint_probes.py
import torch
import os
from typing import Optional, TypeVar

from regnn.constants import TEMP_DIR
from regnn.model import ReGNN
from ..registry import register_probe
from ..dataclass.probe_config import (
    SaveCheckpointProbeScheduleConfig,
    FrequencyType,
)
from ..dataclass.results import CheckpointSavedProbeResult


def save_model(
    model: torch.nn.Module,
    model_type: str = "model",
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    if data_id is not None:
        model_name = os.path.join(save_dir, f"{model_type}_{data_id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"{model_type}_{num_files}.pt")
    torch.save(model.state_dict(), model_name)
    return model_name


@register_probe("save_checkpoint")
def save_checkpoint_probe(
    model: ReGNN,
    schedule_config: SaveCheckpointProbeScheduleConfig,
    epoch: int,
    data_source_name: Optional[str] = "all",
    **kwargs,
) -> Optional[CheckpointSavedProbeResult]:
    current_status = "success"
    status_message = None
    saved_file_path = None

    save_dir = schedule_config.save_dir
    base_name = schedule_config.model_save_name
    file_id_suffix = f"-{schedule_config.file_id}" if schedule_config.file_id else ""

    epoch_part = f"_epoch_{epoch + 1}"
    if schedule_config.frequency_type == FrequencyType.POST_TRAINING:
        epoch_part = "_final"

    data_id_for_save = f"{base_name}{file_id_suffix}{epoch_part}"
    model_type_str = "regnn"

    try:
        saved_file_path = save_model(
            model=model,
            model_type=model_type_str,
            save_dir=save_dir,
            data_id=data_id_for_save,
        )
    except Exception as e:
        import traceback

        current_status = "failure"
        status_message = (
            f"Error in save_checkpoint_probe: {e}\n{traceback.format_exc()}"
        )
        print(status_message)

    result = CheckpointSavedProbeResult(
        data_source=data_source_name or "all",
        file_path=(
            saved_file_path
            if saved_file_path
            else f"failed_to_save_at_{save_dir}/{data_id_for_save}"
        ),
        status=current_status,
        message=status_message,
    )
    return result
