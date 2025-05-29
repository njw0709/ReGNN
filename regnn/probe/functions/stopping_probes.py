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
    Snapshot,
)  # To access historical data
from regnn.probe.dataclass.regression import (
    OLSModeratedResultsProbe,
)  # To check p-values


@register_probe("pval_early_stopping")
def pval_early_stopping_probe(
    schedule_config: PValEarlyStoppingProbeScheduleConfig,
    epoch: int,  # Current epoch, passed by ProbeManager
    shared_resource_accessor: Callable[[str], Optional[Any]],
    # These are standard in probe signature, but may not be used directly by this probe:
    # model: Optional[Any] = None,
    # iteration_in_epoch: Optional[int] = None,
    # global_iteration: Optional[int] = None,
    # dataset: Optional[Any] = None,
    # dataloader: Optional[Any] = None,
    # data_source_name: Optional[str] = None, # This probe checks multiple data sources from history
    # frequency_type: Optional[FrequencyType] = None, # Schedule config has its own frequency
    # training_hp: Optional[Any] = None,
    # model_config: Optional[Any] = None,
    **kwargs: Dict[str, Any],  # Catch-all for any other args
) -> Optional[EarlyStoppingSignalProbeResult]:
    """
    Checks historical OLSModeratedResultsProbe p-values against criteria to signal early stopping.
    Relies on accessing the full Trajectory via shared_resource_accessor("probe_trajectory").
    """

    if epoch < schedule_config.patience:
        # Not yet passed the patience period, so don't check for stopping.
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",  # Redundant due to class default, but explicit
            data_source=(
                schedule_config.data_sources_to_monitor[0].value
                if schedule_config.data_sources_to_monitor
                else DataSource.ALL.value
            ),  # Representational source
            status="skipped",
            message=f"Patience period (epoch {epoch} < {schedule_config.patience}).",
            should_stop=False,
            reason=f"Patience: current epoch {epoch} is less than patience {schedule_config.patience}.",
        )

    trajectory: Optional[Trajectory] = shared_resource_accessor("probe_trajectory")
    if not trajectory:
        # Should not happen if ProbeManager is set up correctly
        print("Warning: PValEarlyStoppingProbe could not access probe_trajectory.")
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=(
                schedule_config.data_sources_to_monitor[0].value
                if schedule_config.data_sources_to_monitor
                else DataSource.ALL.value
            ),
            status="failure",
            message="Probe trajectory not found in shared resources.",
            should_stop=False,
            reason="Configuration error: trajectory not available.",
        )

    # Check results from the last N relevant evaluation points (epochs where RegressionEval ran)
    # We need to find snapshots that correspond to epochs where OLSModeratedResultsProbe would have run.
    # This assumes RegressionEvalProbes run at a certain frequency, and we are looking for
    # schedule_config.n_sequential_epochs_to_pass *consecutive successful evaluations*.

    # Get all snapshots, ordered by time (implicitly by list order in trajectory.data)
    # Filter for snapshots that are within a reasonable lookback window and contain OLSModeratedResultsProbe.
    # The actual evaluation epochs for RegressionEvalProbes might not be every epoch.
    # We need to count consecutive *evaluation instances* where p-values met criteria.

    # Store p-values per monitored data source per epoch found in history
    # Dict[DataSource, Dict[epoch, p_value]]
    p_values_history: Dict[DataSource, Dict[int, float]] = {
        ds: {} for ds in schedule_config.data_sources_to_monitor
    }

    # Iterate through snapshots in reverse to get recent ones first, or just iterate and filter by epoch later
    for snapshot in trajectory.data:  # Snapshots are ordered by execution time
        # We only care about snapshots from EPOCH frequency, as that's when regression evals typically run.
        # This is a simplification; a more robust way is to check if the snapshot contains an OLSModeratedResultsProbe
        # from a RegressionEvalProbeScheduleConfig that matches our interest.
        # For now, we assume RegressionEval runs on EPOCH frequency.
        if snapshot.frequency_context != FrequencyType.EPOCH:
            continue

        snapshot_epoch = snapshot.epoch
        if (
            snapshot_epoch < schedule_config.patience
        ):  # Don't consider epochs before patience
            continue

        for measurement in snapshot.measurements:
            if isinstance(measurement, OLSModeratedResultsProbe):
                # Check if this OLS result is for one of the data sources we are monitoring
                try:
                    # ProbeData.data_source is a string e.g. "TRAIN", "TEST"
                    # schedule_config.data_sources_to_monitor is List[DataSource] (enum members)
                    measurement_ds_enum = DataSource(
                        measurement.data_source
                    )  # Convert string to enum
                    if measurement_ds_enum in schedule_config.data_sources_to_monitor:
                        if (
                            measurement.interaction_pval is not None
                        ):  # Ensure p-value exists
                            # Store the p-value for this data source at this epoch
                            # If multiple OLSModeratedResultsProbe for same source/epoch, last one wins (or take first)
                            p_values_history[measurement_ds_enum][
                                snapshot_epoch
                            ] = measurement.interaction_pval
                except (
                    ValueError
                ):  # If measurement.data_source string is not a valid DataSource enum member
                    print(
                        f"Warning: OLSModeratedResultsProbe had an invalid data_source string: {measurement.data_source}"
                    )
                    continue

    # Now, check for n_sequential_epochs_to_pass criterion
    # We need to find n_sequential_epochs_to_pass where *all* monitored data_sources met the criterion.

    # Get sorted list of unique epochs where we have p-value data for *any* monitored source
    # This helps in iterating through epochs chronologically for sequential check.
    relevant_epochs = sorted(
        list(set(ep for ds_hist in p_values_history.values() for ep in ds_hist.keys()))
    )
    relevant_epochs = [
        ep for ep in relevant_epochs if ep >= schedule_config.patience
    ]  # Ensure patience is respected

    if not relevant_epochs:
        return EarlyStoppingSignalProbeResult(
            probe_type_name="pval_early_stopping",
            data_source=DataSource.ALL.value,  # Representational
            status="skipped",
            message="No relevant OLSModeratedResultsProbe found in history or before patience.",
            should_stop=False,
            reason="No evaluation data to check.",
        )

    sequential_passes = 0
    epochs_passed_criteria = []  # Store epochs that passed

    # Iterate epochs from oldest relevant to newest
    for current_eval_epoch in reversed(
        relevant_epochs
    ):  # Check from most recent backward
        all_monitored_sources_passed_this_epoch = True
        details_for_this_epoch = []

        for monitored_ds in schedule_config.data_sources_to_monitor:
            pval = p_values_history[monitored_ds].get(current_eval_epoch)
            if pval is None:
                # This data source was not evaluated or had no p-value at this specific epoch.
                # This means the sequence is broken for this epoch for *all* sources together.
                all_monitored_sources_passed_this_epoch = False
                details_for_this_epoch.append(
                    f"{monitored_ds.value}: No p-value at epoch {current_eval_epoch}"
                )
                break  # No need to check other data sources for this epoch

            if pval < schedule_config.criterion:
                details_for_this_epoch.append(
                    f"{monitored_ds.value}: pval {pval:.4f} < {schedule_config.criterion} at epoch {current_eval_epoch} - PASSED"
                )
            else:
                details_for_this_epoch.append(
                    f"{monitored_ds.value}: pval {pval:.4f} >= {schedule_config.criterion} at epoch {current_eval_epoch} - FAILED"
                )
                all_monitored_sources_passed_this_epoch = False
                break  # No need to check other data sources for this epoch

        if all_monitored_sources_passed_this_epoch:
            sequential_passes += 1
            epochs_passed_criteria.append(current_eval_epoch)
            if sequential_passes >= schedule_config.n_sequential_epochs_to_pass:
                reason_msg = f"P-value criterion met for {schedule_config.n_sequential_epochs_to_pass} sequential evaluations on {', '.join([ds.value for ds in schedule_config.data_sources_to_monitor])}. Epochs: {sorted(epochs_passed_criteria[:schedule_config.n_sequential_epochs_to_pass])}. Details: {'; '.join(details_for_this_epoch)}."
                print(f"EARLY STOPPING SIGNAL: {reason_msg}")
                return EarlyStoppingSignalProbeResult(
                    probe_type_name="pval_early_stopping",
                    data_source=DataSource.ALL.value,  # Signal is global
                    status="success",
                    message="Early stopping criterion met.",
                    should_stop=True,
                    reason=reason_msg,
                    value=None,  # Could store dict of pvals that triggered
                )
        else:
            # Sequence broken, reset counter if we need strictly consecutive ones from the most recent N.
            # If any epoch fails, the count of *consecutive* passes from the most recent evaluations is broken.
            # The current logic checks if the *last* N evaluations were successful.
            break  # If any of the recent checks fail, we can stop, no need to go further back.

    # If loop finishes without returning, criteria not met for specified sequential passes
    return EarlyStoppingSignalProbeResult(
        probe_type_name="pval_early_stopping",
        data_source=DataSource.ALL.value,
        status="success",  # The probe itself ran successfully
        message=f"Early stopping criterion not met for {schedule_config.n_sequential_epochs_to_pass} sequential evaluations. Current consecutive passes: {sequential_passes}.",
        should_stop=False,
        reason=f"Needed {schedule_config.n_sequential_epochs_to_pass} sequential passes, got {sequential_passes} from recent evaluations.",
    )
