from typing import List, Optional, Tuple, Union, Dict, Any
from torch.utils.data import DataLoader as TorchDataLoader
import torch

from regnn.data import train_test_split, ReGNNDataset
from regnn.model import ReGNN
from regnn.train import KLDLossConfig
from regnn.probe import Trajectory

from regnn.probe.prober import ProbeManager
from regnn.probe.registry import PROBE_REGISTRY
from regnn.probe.dataclass.probe_config import FrequencyType, DataSource
from regnn.probe.dataclass.results import EarlyStoppingSignalProbeResult

from .base import MacroConfig
from .utils import compute_svd, setup_loss_and_optimizer, format_epoch_printout


def train(
    all_dataset: ReGNNDataset,
    macro_config: MacroConfig,
) -> Union[Tuple[ReGNN, Trajectory], ReGNN]:
    """
    Train a ReGNN model using the comprehensive MacroConfig.

    Args:
        macro_config: Configuration object containing all necessary parameters.

    Returns:
        Trained model and optionally the full Trajectory object.
    """
    regnn_model_cfg = macro_config.model
    training_hp = macro_config.training
    probe_opts_main = macro_config.probe

    shared_resources: Dict[str, Any] = {"macro_config": macro_config}
    probe_manager = ProbeManager(
        schedules=probe_opts_main.schedules,
        probe_registry=PROBE_REGISTRY,
        shared_resources=shared_resources,
    )

    # 2. Data Splitting
    train_indices, test_indices = train_test_split(
        num_elems=len(all_dataset),
        train_ratio=training_hp.train_test_split_ratio,
        seed=training_hp.train_test_split_seed,
    )
    train_dataset = all_dataset.get_subset(train_indices)
    test_dataset = (
        all_dataset.get_subset(test_indices) if test_indices is not None else None
    )
    print("train set: ", train_dataset)
    print("test set: ", test_dataset)

    # SVD matrix computation remains the same, using train_dataset.config and regnn_model_cfg
    if regnn_model_cfg.nn_config.svd.enabled:
        moderator_columns = train_dataset.config.moderators
        if isinstance(moderator_columns[0], list):
            svd_matrices = []
            for i, mod_list in enumerate(moderator_columns):
                # Ensure df_orig is used if train_dataset.df might be a view/subset not suitable for direct SVD on all original moderators
                # Assuming train_dataset.df_orig contains the full original data for these columns before any potential subsetting not reflected in config.moderators
                moderators_np = train_dataset.df_orig[mod_list].to_numpy()
                k_dim_svd = (
                    regnn_model_cfg.nn_config.svd.k_dim[i]
                    if isinstance(regnn_model_cfg.svd.k_dim, list)
                    else regnn_model_cfg.nn_config.svd.k_dim
                )
                svd_matrix = compute_svd(moderators_np, k_dim=k_dim_svd)
                svd_matrices.append(svd_matrix)
            regnn_model_cfg.nn_config.svd.svd_matrix = svd_matrices
        else:
            moderators_np = train_dataset.df_orig[moderator_columns].to_numpy()
            k_dim_svd = (
                regnn_model_cfg.nn_config.svd.k_dim[0]
                if isinstance(regnn_model_cfg.nn_config.svd.k_dim, list)
                else regnn_model_cfg.nn_config.svd.k_dim
            )
            regnn_model_cfg.nn_config.svd.svd_matrix = compute_svd(
                moderators_np, k_dim=k_dim_svd
            )

    # 3. Model Initialization using from_config
    model = ReGNN.from_config(
        regnn_model_cfg,
    )
    if training_hp.device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        model.to(torch.device(training_hp.device))

    loss_func_train, regularization_train, optimizer = setup_loss_and_optimizer(
        model, training_hp
    )

    # 4. DataLoader
    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=training_hp.batch_size,
        shuffle=training_hp.shuffle,
    )

    test_dataloader: Optional[TorchDataLoader] = None
    if test_dataset:
        test_dataloader = TorchDataLoader(
            test_dataset,
            batch_size=training_hp.batch_size,
            shuffle=False,
        )

    datasets_map: Dict[DataSource, ReGNNDataset] = {
        DataSource.TRAIN: train_dataset,
        DataSource.ALL: all_dataset,
    }
    if test_dataset:
        datasets_map[DataSource.TEST] = test_dataset

    dataloaders_map: Dict[DataSource, TorchDataLoader] = {
        DataSource.TRAIN: train_dataloader
    }
    if test_dataloader:
        dataloaders_map[DataSource.TEST] = test_dataloader

    # --- PRE-TRAINING PROBES ---
    probe_manager.execute_probes(
        frequency_context=FrequencyType.PRE_TRAINING,
        model=model,
        epoch=-1,
        datasets=datasets_map,
        dataloaders=dataloaders_map,
        training_hp=training_hp,
        model_config=regnn_model_cfg,
    )
    # --- END PRE-TRAINING PROBES ---

    # Training Loop - Refactored for new ProbeManager
    global_iteration_counter = 0
    training_should_continue = True

    for epoch in range(training_hp.epochs):
        if not training_should_continue:
            break

        model.train()
        epoch_train_loss_sum = 0.0
        num_batches_epoch = 0

        for batch_idx, batch_data_cpu in enumerate(train_dataloader):
            # Move batch data to the correct device
            batch_data = {
                k: v.to(training_hp.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data_cpu.items()
            }
            optimizer.zero_grad()

            # Forward pass - model(**input_tensors) if model.forward takes them unpacked
            # ReGNN.forward takes: moderators, focal_predictor, controlled_vars, y (optional), s_weights (optional)
            # So, input_tensors should contain these keys.
            model_input_kwargs = {
                k: batch_data[k]
                for k in ["moderators", "focal_predictor", "controlled_predictors"]
            }
            # targets = torch.squeeze(batch_data["outcome"], -1)
            targets = batch_data["outcome"]
            s_weights = batch_data.get("weights")
            if regnn_model_cfg.use_closed_form_linear_weights:
                model_input_kwargs["y"] = targets
                if s_weights is not None:
                    model_input_kwargs["s_weights"] = s_weights

            predictions = model(**model_input_kwargs)

            batch_loss_main = torch.tensor(0.0, device=training_hp.device)
            if regnn_model_cfg.nn_config.vae and isinstance(
                training_hp.loss_options, KLDLossConfig
            ):
                output_mu, output_log_var = predictions
                batch_loss_main = loss_func_train(
                    output_mu, output_log_var, targets, output_mu
                )
            else:
                batch_loss_main = loss_func_train(predictions, targets)

            batch_loss_reg = torch.tensor(0.0, device=training_hp.device)
            if regularization_train:
                batch_loss_reg = regularization_train(model)

            total_batch_loss = batch_loss_main + batch_loss_reg

            if s_weights is not None:
                # Ensure s_weights is 1D and matches batch_loss_main's batch dimension if batch_loss_main is not scalar
                if (
                    total_batch_loss.ndim > 0
                    and total_batch_loss.shape[0] == s_weights.shape[0]
                ):
                    total_batch_loss = (
                        total_batch_loss * s_weights
                    ).sum() / s_weights.sum()
                elif total_batch_loss.ndim == 0:  # If loss is already scalar
                    pass  # Assuming scalar loss doesn't need further weighting here if already handled or not applicable
                else:
                    # This case might require broadcasting or careful handling
                    # For now, default to simple mean if shapes don't align for direct weighting as above
                    print(
                        f"Warning: Survey weight application mismatch. Loss dim: {total_batch_loss.ndim}, S_weights dim: {s_weights.ndim}. Using unweighted mean."
                    )
                    total_batch_loss = (
                        total_batch_loss.mean()
                        if total_batch_loss.ndim > 0
                        else total_batch_loss
                    )
            elif (
                total_batch_loss.ndim > 0
            ):  # Default mean reduction if not using survey weights and loss is not scalar
                total_batch_loss = total_batch_loss.mean()

            total_batch_loss.backward()
            optimizer.step()

            epoch_train_loss_sum += (
                total_batch_loss.item()
            )  # .item() gets scalar from 0-dim tensor
            num_batches_epoch += 1
            global_iteration_counter += 1

            # --- ITERATION-LEVEL PROBES ---
            iteration_probes_results = probe_manager.execute_probes(
                frequency_context=FrequencyType.ITERATION,
                model=model,
                epoch=epoch,
                iteration_in_epoch=batch_idx,
                global_iteration=global_iteration_counter,
                datasets=datasets_map,
                dataloaders=dataloaders_map,
                training_hp=training_hp,
                model_config=regnn_model_cfg,
            )
            # Iteration-level early stopping (less common, but possible)
            for res in iteration_probes_results:
                if isinstance(res, EarlyStoppingSignalProbeResult) and res.should_stop:
                    print(
                        f"Early stopping triggered at ITERATION level: Epoch {epoch}, Iteration {batch_idx}. Reason: {res.reason or res.message}"
                    )
                    training_should_continue = False
                    break
            if not training_should_continue:
                break  # Break from batch loop
        if not training_should_continue:
            break  # Break from epoch loop

        avg_epoch_train_loss = (
            epoch_train_loss_sum / num_batches_epoch
            if num_batches_epoch > 0
            else float("nan")
        )
        base_printout = f"Epoch {epoch + 1}/{training_hp.epochs} | Avg Train Batch Loss: {avg_epoch_train_loss:.6f}"

        # Store avg epoch train loss for objective_probe pre-computation
        # The breakdown here is simplified; a more accurate reg loss would require summing batch_loss_reg
        # or the objective_probe for train can re-calculate with its own loop if perfect breakdown is needed.
        current_epoch_objective_breakdown = {
            "main_loss": avg_epoch_train_loss,  # This is total loss, not just main if reg is applied
            # "regularization_loss": avg_epoch_reg_loss (if tracked separately per epoch)
        }
        shared_resources[f"epoch_objective_{DataSource.TRAIN.value}_e{epoch}"] = (
            avg_epoch_train_loss,
            f"total_loss_on_{DataSource.TRAIN.value}",
            current_epoch_objective_breakdown,
        )

        # --- EPOCH-LEVEL PROBES ---
        # Note: Test loss for printout will now come from an ObjectiveProbe result if scheduled.
        epoch_probes_results = probe_manager.execute_probes(
            frequency_context=FrequencyType.EPOCH,
            model=model,
            epoch=epoch,
            iteration_in_epoch=None,  # Not relevant for epoch freq
            global_iteration=global_iteration_counter,  # Pass global iter if needed by any epoch probe
            datasets=datasets_map,
            dataloaders=dataloaders_map,
            training_hp=training_hp,
            model_config=regnn_model_cfg,
        )
        # Check for early stopping signal from epoch-level probes
        for res in epoch_probes_results:
            if isinstance(res, EarlyStoppingSignalProbeResult) and res.should_stop:
                print(
                    f"Early stopping signaled at EPOCH level: Epoch {epoch}. Reason: {res.reason or res.message}"
                )
                training_should_continue = (
                    False  # Signal to stop training after this epoch's processing
                )
                break  # Stop checking other probe results for this epoch
        # The loop will break at the start of the next epoch if training_should_continue is False

        # Format and print epoch results including probe data
        epoch_printout = format_epoch_printout(base_printout, epoch_probes_results)
        print(epoch_printout)

    # --- POST-TRAINING PROBES ---
    # Run regardless of early stopping, but after the loop has finished or been broken
    # The epoch number for post-training could be the last completed epoch or training_hp.epochs
    final_epoch_for_post_probes = (
        epoch
        if not training_should_continue and epoch < training_hp.epochs - 1
        else training_hp.epochs - 1
    )

    probe_manager.execute_probes(
        frequency_context=FrequencyType.POST_TRAINING,
        model=model,
        epoch=final_epoch_for_post_probes,  # Use the actual last epoch or configured total for context
        iteration_in_epoch=None,
        global_iteration=global_iteration_counter,  # Final global iteration count
        datasets=datasets_map,
        dataloaders=dataloaders_map,
        training_hp=training_hp,
        model_config=regnn_model_cfg,  # model.config
    )

    if probe_opts_main.return_trajectory:
        # Return the history of all ProbeData collected by the ProbeManager
        return model, probe_manager.trajectory

    return model
