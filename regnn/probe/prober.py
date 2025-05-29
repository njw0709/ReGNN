from typing import List, Dict, Callable, Any, Optional, Union, Type, TypeVar

# Typing for Pydantic models and Enums
from .dataclass.probe_config import (
    ProbeScheduleConfig,
    FrequencyType,
    DataSource,
    RegressionEvalProbeScheduleConfig,
    SaveCheckpointProbeScheduleConfig,
    SaveIntermediateIndexProbeScheduleConfig,
    GetObjectiveProbeScheduleConfig,
    GetL2LengthProbeScheduleConfig,
)

from regnn.model import ReGNN
from torch.utils.data import Dataset, DataLoader
from regnn.probe import ProbeData


class ProbeManager:
    def __init__(
        self,
        schedules: List[
            Union[
                RegressionEvalProbeScheduleConfig,
                SaveCheckpointProbeScheduleConfig,
                SaveIntermediateIndexProbeScheduleConfig,
                GetObjectiveProbeScheduleConfig,
                GetL2LengthProbeScheduleConfig,
                ProbeScheduleConfig,
            ]
        ],  # Explicit Union type for schedules
        probe_registry: Dict[str, Callable[..., Optional[ProbeData]]],
        shared_resources: Optional[Dict[str, Any]] = None,
    ):
        self.schedules = schedules
        self.probe_registry = probe_registry
        self.shared_resources = shared_resources if shared_resources is not None else {}
        self.results_history: List[ProbeData] = []  # To store results from probes

    def get_shared_resource(self, key: str) -> Optional[Any]:
        """Safely retrieves a shared resource by its key."""
        return self.shared_resources.get(key)

    def execute_probes(
        self,
        frequency_context: FrequencyType,
        # --- Contextual arguments ---
        model: ReGNN,
        epoch: int = -1,  # -1 could indicate pre/post training, or iteration 0 of epoch 0
        iteration_in_epoch: Optional[int] = None,  # For ITERATION frequency
        global_iteration: Optional[int] = None,  # For ITERATION frequency
        # Data related context (provide all available standard datasets/loaders)
        datasets: Optional[Dict[DataSource, Dataset]] = None,
        dataloaders: Optional[Dict[DataSource, DataLoader]] = None,
        # Pass other relevant parts of macro_config if probes need them
        training_hp: Optional[Any] = None,  # Example: macro_config.training
        model_config: Optional[Any] = None,  # Example: macro_config.model
        # **kwargs for any other dynamic context
    ) -> List[
        ProbeData
    ]:  # Returns a list of probe data collected during this execution call
        """
        Executes all probes that are due based on the current context.
        """
        collected_probe_data: List[ProbeData] = []
        active_datasets = datasets if datasets is not None else {}
        active_dataloaders = dataloaders if dataloaders is not None else {}

        for schedule in self.schedules:
            if schedule.frequency_type != frequency_context:
                continue

            # Check frequency_value for EPOCH and ITERATION types
            due_this_run = False
            if (
                frequency_context == FrequencyType.PRE_TRAINING
                or frequency_context == FrequencyType.POST_TRAINING
            ):
                due_this_run = True
            elif frequency_context == FrequencyType.EPOCH:
                if (epoch + 1) % schedule.frequency_value == 0:  # epoch is 0-indexed
                    due_this_run = True
            elif (
                frequency_context == FrequencyType.ITERATION
                and iteration_in_epoch is not None
            ):
                if (iteration_in_epoch + 1) % schedule.frequency_value == 0:
                    due_this_run = True

            if not due_this_run:
                continue

            probe_func = self.probe_registry.get(schedule.probe_type)
            if not probe_func:
                print(
                    f"Warning: Probe type '{schedule.probe_type}' not found in registry. Skipping."
                )
                continue

            for ds_enum in schedule.data_sources:
                current_dataset: Optional[Dataset] = active_datasets.get(ds_enum)
                current_dataloader: Optional[DataLoader] = active_dataloaders.get(
                    ds_enum
                )

                if ds_enum == DataSource.ALL:
                    pass  # Probe handles None dataset/dataloader if it's truly global
                elif current_dataset is None and current_dataloader is None:
                    print(
                        f"Warning: No dataset/dataloader for data_source '{ds_enum.value}' for probe '{schedule.probe_type}'. Skipping this data_source for this probe."
                    )
                    continue

                print(
                    f"Executing probe: {schedule.probe_type} on {ds_enum.value} (Epoch: {epoch}, Iter: {iteration_in_epoch})"
                )
                try:
                    probe_kwargs = {
                        "model": model,
                        "epoch": epoch,
                        "iteration_in_epoch": iteration_in_epoch,
                        "global_iteration": global_iteration,
                        "dataset": current_dataset,
                        "dataloader": current_dataloader,
                        "data_source_name": ds_enum.value,
                        "schedule_config": schedule,
                        "probe_params": schedule.probe_params or {},
                        "shared_resource_accessor": self.get_shared_resource,
                        "training_hp": training_hp,
                        "model_config": model_config,
                    }

                    probe_result = probe_func(**probe_kwargs)

                    if probe_result is not None:
                        if isinstance(probe_result, list):
                            collected_probe_data.extend(probe_result)
                            self.results_history.extend(probe_result)
                        else:
                            collected_probe_data.append(probe_result)
                            self.results_history.append(probe_result)
                except Exception as e:
                    print(
                        f"Error executing probe '{schedule.probe_type}' on data_source '{ds_enum.value}': {e}"
                    )
                    import traceback

                    traceback.print_exc()

        return collected_probe_data
