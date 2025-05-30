from .dataclass import (
    ProbeData,
    ProbeOptions,
    Snapshot,
    Trajectory,
    ObjectiveProbe,
    ModeratedRegressionConfig,
    OLSResultsProbe,
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
    L2NormProbe,
    CheckpointSavedProbeResult,
    IntermediateIndexSavedProbeResult,
    FrequencyType,
    DataSource,
    ProbeScheduleConfig,
    RegressionEvalProbeScheduleConfig,
    SaveIntermediateIndexProbeScheduleConfig,
    SaveCheckpointProbeScheduleConfig,
    GetObjectiveProbeScheduleConfig,
    GetL2LengthProbeScheduleConfig,
    PValEarlyStoppingProbeScheduleConfig,
    FocalPredictorPreProcessOptions,
)
from .registry import PROBE_REGISTRY, register_probe

# Import functions to ensure registration
from .functions import (
    save_checkpoint_probe,
    save_intermediate_index_probe,
    l2_length_probe,
    objective_probe,
    regression_eval_probe,
    generate_stata_command,
    pval_early_stopping_probe,
)  # This will execute functions/__init__.py

__all__ = [
    # dataclass exports - ProbeData and specific result types
    "ProbeData",
    "ProbeOptions",
    "Snapshot",
    "Trajectory",
    "ObjectiveProbe",
    "OLSResultsProbe",
    "ModeratedRegressionConfig",
    "OLSModeratedResultsProbe",
    "VarianceInflationFactorProbe",
    "L2NormProbe",
    "CheckpointSavedProbeResult",
    "IntermediateIndexSavedProbeResult",
    # dataclass exports - Probe Configurations
    "FrequencyType",
    "DataSource",
    "ProbeScheduleConfig",
    "RegressionEvalProbeScheduleConfig",
    "SaveCheckpointProbeScheduleConfig",
    "SaveIntermediateIndexProbeScheduleConfig",
    "GetObjectiveProbeScheduleConfig",
    "GetL2LengthProbeScheduleConfig",
    "PValEarlyStoppingProbeScheduleConfig",
    "FocalPredictorPreProcessOptions",
    # registry exports
    "PROBE_REGISTRY",
    "register_probe",
    # functions module (optional to export, but importing it is key for registration)
    "save_checkpoint_probe",
    "save_intermediate_index_probe",
    "l2_length_probe",
    "objective_probe",
    "regression_eval_probe",
    "generate_stata_command",
    "pval_early_stopping_probe",
]
