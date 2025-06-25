from typing import Optional, List, Union, Callable, Any, Dict

from regnn.probe.registry import register_probe, PROBE_REGISTRY
from regnn.probe.dataclass.base import ProbeData
from regnn.probe.dataclass.probe_config import (
    PValEarlyStoppingProbeScheduleConfig,
    DataSource,  # For type hinting
    FrequencyType,  # For type hinting
)
from regnn.probe.dataclass.results import EarlyStoppingSignalProbeResult
from regnn.probe.dataclass.trajectory import (
    Trajectory,
)  # To access historical data
from regnn.probe.dataclass.regression import (
    OLSModeratedResultsProbe,
)  # To check p-values


@register_probe("pval_early_stopping")
def pval_early_stopping_probe(
    schedule_config: PValEarlyStoppingProbeScheduleConfig,
    epoch: int,
    shared_resource_accessor: Callable[[str], Optional[Any]],
    **kwargs: Dict[str, Any],  # Catch-all for any other args
) -> Optional[EarlyStoppingSignalProbeResult]:
    """
    Checks historical OLSModeratedResultsProbe p-values against criteria to signal early stopping.
    Relies on accessing the full Trajectory via shared_resource_accessor("probe_trajectory").
    Considers the N most recent joint evaluation epochs where all monitored data sources have p-values.
    """

    if not schedule_config.data_sources_to_monitor:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL,
            status="skipped",
            message="No data sources configured for monitoring.",
            should_stop=False,
            reason="Configuration error: data_sources_to_monitor is empty.",
        )

    if epoch < schedule_config.patience:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=(
                schedule_config.data_sources_to_monitor[0]
                if schedule_config.data_sources_to_monitor
                else DataSource.ALL
            ),
            status="skipped",
            message=f"Patience period not met (current epoch {epoch} < patience {schedule_config.patience}).",
            should_stop=False,
            reason=f"Patience: current epoch {epoch} is less than patience {schedule_config.patience}.",
        )

    trajectory: Optional[Trajectory] = shared_resource_accessor("probe_trajectory")
    if not trajectory:
        print("Warning: PValEarlyStoppingProbe could not access probe_trajectory.")
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=(
                schedule_config.data_sources_to_monitor[0]
                if schedule_config.data_sources_to_monitor
                else DataSource.ALL
            ),
            status="failure",
            message="Probe trajectory not found in shared resources.",
            should_stop=False,
            reason="Configuration error: trajectory not available.",
        )

    p_values_history: Dict[DataSource, Dict[int, float]] = {
        ds: {} for ds in schedule_config.data_sources_to_monitor
    }

    # Group snapshots by epoch and frequency type
    epoch_snapshots: Dict[int, List[Any]] = {}
    iteration_snapshots: Dict[int, List[Any]] = {}

    for snapshot in trajectory.data:
        snapshot_epoch = snapshot.epoch

        if snapshot.frequency_context == FrequencyType.EPOCH:
            if snapshot_epoch not in epoch_snapshots:
                epoch_snapshots[snapshot_epoch] = []
            epoch_snapshots[snapshot_epoch].append(snapshot)
        elif snapshot.frequency_context == FrequencyType.ITERATION:
            if snapshot_epoch not in iteration_snapshots:
                iteration_snapshots[snapshot_epoch] = []
            iteration_snapshots[snapshot_epoch].append(snapshot)

    def process_snapshot_measurements(snapshot, snapshot_epoch):
        """Helper function to process measurements from a snapshot"""
        for measurement in snapshot.measurements:
            if isinstance(measurement, OLSModeratedResultsProbe):
                try:
                    measurement_ds_enum = DataSource(measurement.data_source)
                    if measurement_ds_enum in schedule_config.data_sources_to_monitor:
                        if measurement.interaction_pval is not None:
                            p_values_history[measurement_ds_enum][
                                snapshot_epoch
                            ] = measurement.interaction_pval
                except ValueError:
                    print(
                        f"Warning: OLSModeratedResultsProbe had an invalid data_source string: {measurement.data_source}"
                    )
                    continue

    # Process epoch-based snapshots first
    for epoch, snapshots in epoch_snapshots.items():
        for snapshot in snapshots:
            process_snapshot_measurements(snapshot, epoch)

    # Process iteration-based snapshots (take last iteration per epoch)
    # Only for epochs that don't already have epoch-based measurements
    for epoch, snapshots in iteration_snapshots.items():
        if epoch not in epoch_snapshots:
            # Take the last snapshot for this epoch (assuming chronological order)
            last_snapshot = snapshots[-1]
            process_snapshot_measurements(last_snapshot, epoch)

    # Identify joint evaluation epochs where ALL monitored sources have p-values and epoch >= patience
    all_epochs_with_any_data = sorted(
        list(set(ep for ds_hist in p_values_history.values() for ep in ds_hist.keys()))
    )
    joint_evaluation_epochs = []
    for e in all_epochs_with_any_data:
        if e < schedule_config.patience:
            continue
        is_joint = True
        for monitored_ds in schedule_config.data_sources_to_monitor:
            if e not in p_values_history[monitored_ds]:
                is_joint = False
                break
        if is_joint:
            joint_evaluation_epochs.append(e)

    if not joint_evaluation_epochs:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL,
            status="skipped",
            message="No joint evaluation epochs found in history where all monitored data sources have p-values (or none after patience period).",
            should_stop=False,
            reason="No suitable historical evaluation data to check.",
        )

    if len(joint_evaluation_epochs) < schedule_config.n_sequential_evals_to_pass:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL,
            status="skipped",
            message=(
                f"Not enough joint evaluation epochs ({len(joint_evaluation_epochs)}) to check for "
                f"{schedule_config.n_sequential_evals_to_pass} sequential passes. "
                f"Patience is {schedule_config.patience}."
            ),
            should_stop=False,
            reason=f"Requires {schedule_config.n_sequential_evals_to_pass} joint evaluations, found {len(joint_evaluation_epochs)} after patience.",
        )

    # Examine the N most recent joint evaluation epochs
    epochs_to_check = joint_evaluation_epochs[
        -schedule_config.n_sequential_evals_to_pass :
    ]

    all_n_recent_passed = True
    failure_reason_detail = ""
    passed_epochs_details_list = []

    for (
        current_eval_epoch
    ) in epochs_to_check:  # These are already sorted and filtered by patience
        current_epoch_all_ds_passed = True
        details_for_this_epoch_pass = []

        for monitored_ds in schedule_config.data_sources_to_monitor:
            # P-value must exist here because these are joint_evaluation_epochs
            pval = p_values_history[monitored_ds][current_eval_epoch]

            if pval < schedule_config.criterion:
                details_for_this_epoch_pass.append(
                    f"{monitored_ds}: pval {pval:.4f} < {schedule_config.criterion} at epoch {current_eval_epoch} - PASSED"
                )
            else:
                failure_reason_detail = f"Epoch {current_eval_epoch}, {monitored_ds}: pval {pval:.4f} >= {schedule_config.criterion} - FAILED."
                current_epoch_all_ds_passed = False
                all_n_recent_passed = False
                break  # This data source failed, so this epoch fails

        if current_epoch_all_ds_passed:
            passed_epochs_details_list.append(
                f"Epoch {current_eval_epoch} ({'; '.join(details_for_this_epoch_pass)})"
            )

        if not all_n_recent_passed:
            break  # One of the N recent epochs failed, no need to check further

    if all_n_recent_passed:
        reason_msg = (
            f"P-value criterion met for the {schedule_config.n_sequential_evals_to_pass} most recent joint evaluations "
            f"on {', '.join([ds for ds in schedule_config.data_sources_to_monitor])}. "
            f"Evaluated epochs: {sorted(epochs_to_check)}. Details: {'; '.join(passed_epochs_details_list)}."
        )
        print(f"EARLY STOPPING SIGNAL: {reason_msg}")
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL,
            status="success",
            message="Early stopping criterion met based on N most recent joint evaluations.",
            should_stop=True,
            reason=reason_msg,
        )
    else:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL,
            status="success",  # Probe ran successfully
            message=(
                f"Early stopping criterion NOT met for the {schedule_config.n_sequential_evals_to_pass} most recent joint evaluations. "
                f"{failure_reason_detail}"
            ),
            should_stop=False,
            reason=(
                f"Needed {schedule_config.n_sequential_evals_to_pass} most recent joint evaluations to pass. "
                f"{failure_reason_detail}"
            ),
        )
