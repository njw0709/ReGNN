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
