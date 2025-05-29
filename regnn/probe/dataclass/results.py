from typing import Optional, Any, List, Dict
from pydantic import Field
from .base import ProbeData


class CheckpointSavedProbeResult(ProbeData):
    """Result of a checkpoint saving operation."""

    # probe_type_name will be "CheckpointSavedProbeResult" via inheritance
    file_path: str = Field(..., description="Path to the saved checkpoint file.")
    # All other fields (epoch, timestamp, status, etc.) are inherited from base ProbeData (which is now leaner)
    # Snapshot will provide epoch, time context.


class IntermediateIndexSavedProbeResult(ProbeData):
    """Result of saving intermediate index predictions."""

    # probe_type_name will be "IntermediateIndexSavedProbeResult"
    file_path: str = Field(
        ..., description="Path to the saved intermediate index file (e.g., .dta)."
    )


# --- EarlyStoppingSignalProbeResult ---
class EarlyStoppingSignalProbeResult(ProbeData):
    probe_type_name: str = Field("early_stopping_signal", const=True)
    should_stop: bool = Field(..., description="Indicates if the training should stop.")
    reason: Optional[str] = Field(None, description="Reason for the stopping signal.")
    # You can add more fields here if needed, e.g., the epoch it was triggered, specific p-values etc.
    # value: Optional[Any] = Field(None, description="Can store the specific value that triggered stopping, e.g., p-value")
