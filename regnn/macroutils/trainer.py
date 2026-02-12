from typing import List, Optional, Tuple, Union, Dict, Any
from torch.utils.data import DataLoader as TorchDataLoader
import torch

from regnn.data import train_test_split, ReGNNDataset, kfold_split, train_test_val_split
from regnn.model import ReGNN
from regnn.train import KLDLossConfig, TreeLossConfig, PriorPenaltyLossConfig
from regnn.probe import Trajectory

from regnn.probe.prober import ProbeManager
from regnn.probe.registry import PROBE_REGISTRY
from regnn.probe.dataclass.probe_config import FrequencyType, DataSource
from regnn.probe.dataclass.results import EarlyStoppingSignalProbeResult

from .base import MacroConfig
from .utils import (
    setup_loss_and_optimizer,
    format_epoch_printout,
    balance_gradients_for_regnn,
)


def train(
    all_dataset: ReGNNDataset,
    macro_config: MacroConfig,
    gradient_balance: float = -1,
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
    if training_hp.kfold is not None:
        train_indices, test_indices = kfold_split(
            num_elems=len(all_dataset),
            k=training_hp.kfold,
            holdout_fold=training_hp.k_to_hold,
            seed=training_hp.train_test_split_seed,
        )
        if training_hp.val_ratio > 0.0:
            val_set_num = int(len(all_dataset) * training_hp.val_ratio)
            print(val_set_num)
            if val_set_num < len(test_indices):
                test_indices = test_indices[:val_set_num]
    elif training_hp.val_ratio > 0.0:
        train_indices, _, test_indices = train_test_val_split(
            num_elems=len(all_dataset),
            train_ratio=training_hp.train_test_split_ratio,
            test_ratio=1 - (training_hp.train_test_split_ratio + training_hp.val_ratio),
            val_ratio=training_hp.val_ratio,
            seed=training_hp.train_test_split_seed,
        )
    else:
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

    # 3. Model Initialization using from_config
    model = ReGNN.from_config(
        regnn_model_cfg,
    )
    if training_hp.device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        model.to(torch.device(training_hp.device))

    loss_func_train, regularization_train, optimizer, scheduler = (
        setup_loss_and_optimizer(model, training_hp)
    )

    # 4. DataLoader
    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=training_hp.batch_size,
        shuffle=training_hp.shuffle,
        drop_last=True,
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

    # Expose datasets and training_hp to probes via shared resources
    # (needed by save-on-best in pval early stopping)
    shared_resources["datasets_map"] = datasets_map
    shared_resources["training_hp"] = training_hp

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

    # Initialize temperature annealer if configured
    temperature_annealer = None
    if (
        training_hp.optimizer_config.temperature_annealing is not None
        and regnn_model_cfg.nn_config.use_soft_tree
    ):
        from regnn.macroutils.utils import TemperatureAnnealer

        temperature_annealer = TemperatureAnnealer(
            training_hp.optimizer_config.temperature_annealing, training_hp.epochs
        )

    # Initialize batch size scheduler if configured
    batch_size_scheduler = None
    if training_hp.batch_size_scheduler is not None:
        from regnn.macroutils.utils import BatchSizeScheduler

        batch_size_scheduler = BatchSizeScheduler(
            training_hp.batch_size_scheduler, training_hp.batch_size
        )

    # Setup regression gradient accumulation (sample-based)
    reg_accum_target = training_hp.regression_gradient_accumulation_samples
    use_reg_accum = reg_accum_target > 1
    if use_reg_accum:
        # Initialize accumulator for regression (MMR) parameter gradients
        reg_grad_accum = {
            id(p): torch.zeros_like(p, device=training_hp.device)
            for p in model.mmr_parameters
        }
        reg_accum_samples_count = 0  # tracks accumulated samples
        reg_accum_batch_count = 0  # tracks accumulated batches (for gradient averaging)
        print(
            f"Regression gradient accumulation enabled: "
            f"regression params update every {reg_accum_target} samples"
        )

    # Training Loop - Refactored for new ProbeManager
    global_iteration_counter = 0
    training_should_continue = True

    for epoch in range(training_hp.epochs):
        if not training_should_continue:
            break

        # Update batch size if scheduled
        if batch_size_scheduler:
            new_batch_size, should_recreate = batch_size_scheduler.step(epoch)
            if should_recreate:
                print(
                    f"Epoch {epoch}: Updating batch size from {train_dataloader.batch_size} to {new_batch_size}"
                )
                train_dataloader = TorchDataLoader(
                    train_dataset,
                    batch_size=new_batch_size,
                    shuffle=training_hp.shuffle,
                    drop_last=True,
                )
                dataloaders_map[DataSource.TRAIN] = train_dataloader

        # Update temperature for SoftTree models
        if temperature_annealer:
            temperature_annealer.update_model_temperature(model, epoch)

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
                outcome, mu, log_var = predictions
                batch_loss_main = loss_func_train(targets, outcome, mu, log_var)
            elif isinstance(training_hp.loss_options, TreeLossConfig):
                # Tree routing regularization requires moderators and model
                moderators = batch_data["moderators"]
                batch_loss_main = loss_func_train(
                    predictions, targets, moderators, model
                )
            elif isinstance(training_hp.loss_options, PriorPenaltyLossConfig):
                # Prior penalty requires moderators and model
                moderators = batch_data["moderators"]
                batch_loss_main = loss_func_train(
                    predictions, targets, moderators, model
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

            # Regression gradient accumulation (sample-based): accumulate
            # regression gradients until enough samples have been seen, while
            # NN params update every step.
            if use_reg_accum:
                for p in model.mmr_parameters:
                    if p.grad is not None:
                        reg_grad_accum[id(p)] += p.grad.clone()
                current_batch_size = targets.shape[0]
                reg_accum_samples_count += current_batch_size
                reg_accum_batch_count += 1

                if reg_accum_samples_count < reg_accum_target:
                    # Not ready to update regression yet -- zero their grads
                    # so optimizer.step() only updates NN params
                    for p in model.mmr_parameters:
                        if p.grad is not None:
                            p.grad.zero_()
                else:
                    # Ready to update regression -- replace grads with averaged accumulation
                    for p in model.mmr_parameters:
                        if p.grad is not None:
                            p.grad.copy_(reg_grad_accum[id(p)] / reg_accum_batch_count)
                    # Reset accumulators
                    for key in reg_grad_accum:
                        reg_grad_accum[key].zero_()
                    reg_accum_samples_count = 0
                    reg_accum_batch_count = 0

            if gradient_balance > 0:
                balance_gradients_for_regnn(model, desired_ratio=gradient_balance)
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
                        f"Early stopping triggered at iteration level: Epoch {epoch} (1-indexed: epoch {epoch + 1}), "
                        f"Iteration {batch_idx}. Training will stop. Reason: {res.reason or res.message}"
                    )
                    training_should_continue = False
                    break
            if not training_should_continue:
                break  # Break from batch loop
        # Flush any remaining accumulated regression gradients at end of epoch
        if use_reg_accum and reg_accum_batch_count > 0:
            # Apply partial accumulation so regression params don't miss end-of-epoch data
            optimizer.zero_grad()
            # We need a dummy backward to populate NN grads (they'll be zero).
            # Just directly set regression grads from accumulator.
            for p in model.mmr_parameters:
                if reg_grad_accum[id(p)] is not None:
                    p.grad = (reg_grad_accum[id(p)] / reg_accum_batch_count).clone()
            optimizer.step()
            # Reset accumulators for next epoch
            for key in reg_grad_accum:
                reg_grad_accum[key].zero_()
            reg_accum_samples_count = 0
            reg_accum_batch_count = 0

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
                    f"Early stopping signaled after epoch {epoch} (1-indexed: epoch {epoch + 1}). "
                    f"Training will stop before the next epoch. Reason: {res.reason or res.message}"
                )
                training_should_continue = (
                    False  # Signal to stop training before the next epoch starts
                )
                break  # Stop checking other probe results for this epoch
        # The loop will break at the start of the next epoch if training_should_continue is False

        # Format and print epoch results including probe data
        epoch_printout = format_epoch_printout(base_printout, epoch, probe_manager)
        print(epoch_printout)

        # Update learning rate scheduler
        if scheduler:
            from regnn.macroutils.utils import PerGroupScheduler

            # Check if this is a per-group scheduler or standard scheduler
            is_per_group = isinstance(scheduler, PerGroupScheduler)
            is_reduce_on_plateau = (
                is_per_group and scheduler.is_reduce_on_plateau
            ) or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

            if is_reduce_on_plateau:
                # ReduceLROnPlateau needs a metric to monitor
                # Try to use test loss if available, otherwise use train loss
                metric_value = avg_epoch_train_loss
                # Check if test objective was computed in epoch probes
                for res in epoch_probes_results:
                    from regnn.probe import ObjectiveProbe

                    if isinstance(res, ObjectiveProbe) and res.data_source == "test":
                        metric_value = res.objective
                        break
                scheduler.step(metric_value)
            else:
                scheduler.step()

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
