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
from .dataclass.base import ProbeData as BaseProbeData
from .dataclass.trajectory import Snapshot, Trajectory


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
        probe_registry: Dict[
            str, Callable[..., Optional[Union[ProbeData, List[ProbeData]]]]
        ],
        shared_resources: Optional[Dict[str, Any]] = None,
    ):
        self.schedules = schedules
        self.probe_registry = probe_registry
        self.shared_resources = shared_resources if shared_resources is not None else {}
        self.trajectory = Trajectory()  # Initialize Trajectory

        # Make the trajectory itself available via shared resources for probes like early stopping
        self.shared_resources["probe_trajectory"] = self.trajectory

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
        current_snapshot = Snapshot(
            epoch=epoch,
            iteration_in_epoch=iteration_in_epoch,
            global_iteration=global_iteration,
            frequency_context=frequency_context,
            measurements=[],
        )
        collected_probe_data_for_this_call: List[ProbeData] = []
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
                error_probe_data = ProbeData(
                    data_source=DataSource.ALL,
                    status="error",
                    message=f"Probe type '{schedule.probe_type}' not found",
                    probe_type_name="unknown_probe_error",
                )
                current_snapshot.add(error_probe_data)
                collected_probe_data_for_this_call.append(error_probe_data)
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
                        f"Warning: No dataset/dataloader for data_source '{ds_enum}' for probe '{schedule.probe_type}'. Skipping this data_source for this probe."
                    )
                    skipped_probe_data = ProbeData(
                        data_source=ds_enum,
                        status="skipped",
                        message=f"Dataset for {ds_enum} not provided",
                        probe_type_name=schedule.probe_type,
                    )
                    current_snapshot.add(skipped_probe_data)
                    collected_probe_data_for_this_call.append(skipped_probe_data)
                    continue

                print(
                    f"Executing probe: {schedule.probe_type} on {ds_enum} (Epoch: {epoch}, Iter: {iteration_in_epoch})"
                )
                try:
                    probe_kwargs = {
                        "model": model,
                        "epoch": epoch,
                        "iteration_in_epoch": iteration_in_epoch,
                        "global_iteration": global_iteration,
                        "dataset": current_dataset,
                        "dataloader": current_dataloader,
                        "data_source_name": ds_enum,
                        "schedule_config": schedule,
                        "frequency_type": frequency_context,
                        "shared_resource_accessor": self.get_shared_resource,
                        "training_hp": training_hp,
                        "model_config": model_config,
                    }

                    probe_result = probe_func(**probe_kwargs)

                    if probe_result is not None:
                        results_to_add = (
                            probe_result
                            if isinstance(probe_result, list)
                            else [probe_result]
                        )
                        for res_item in results_to_add:
                            if isinstance(res_item, ProbeData):
                                current_snapshot.add(res_item)
                                collected_probe_data_for_this_call.append(res_item)
                            else:
                                print(
                                    f"Warning: Probe '{schedule.probe_type}' returned an item not instance of ProbeData: {type(res_item)}. Skipping this item."
                                )
                                error_data = ProbeData(
                                    data_source=ds_enum,
                                    status="error",
                                    message=f"Probe {schedule.probe_type} returned non-ProbeData: {type(res_item)}",
                                    probe_type_name=schedule.probe_type,
                                )
                                current_snapshot.add(error_data)
                                collected_probe_data_for_this_call.append(error_data)
                except Exception as e:
                    print(
                        f"Error executing probe '{schedule.probe_type}' on data_source '{ds_enum}': {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    error_probe_data = ProbeData(
                        data_source=ds_enum,
                        status="failure",
                        message=str(e),
                        probe_type_name=schedule.probe_type,
                    )
                    current_snapshot.add(error_probe_data)
                    collected_probe_data_for_this_call.append(error_probe_data)

        if current_snapshot.measurements:
            self.trajectory.append(current_snapshot)

        return collected_probe_data_for_this_call
