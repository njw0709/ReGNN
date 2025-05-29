# Placeholder for __init__.py in functions directory
# This file will be used to import probe functions to ensure they are registered.

# Import all probe functions here to ensure they are registered with the PROBE_REGISTRY

from .checkpoint_probes import save_checkpoint_probe
from .intermediate_index_probes import save_intermediate_index_probe
from .l2_probes import l2_length_probe
from .objective_probes import objective_probe
from .regression_eval_probes import regression_eval_probe
from .stopping_probes import pval_early_stopping_probe

# from . import custom_probes # Example for other probe categories

__all__ = [
    "save_checkpoint_probe",
    "save_intermediate_index_probe",
    "l2_length_probe",
    "objective_probe",
    "regression_eval_probe",
    "pval_early_stopping_probe",
]
