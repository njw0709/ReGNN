import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, List, Callable, Union, Type

from regnn.probe import Snapshot, ProbeData, Trajectory, ObjectiveProbe, L2NormProbe
from regnn.model import ReGNN


def _compute_loss(
    loss_function: nn.Module,
    outputs: Any,
    target: torch.Tensor,
    vae_loss_active: bool,
) -> torch.Tensor:
    if vae_loss_active:
        # outputs is expected to be a tuple: (y_pred_mean, int_pred_mu, int_pred_logvar)
        # target is sample["outcome"]
        # vae_kld_regularized_loss expects: (y_true, y_pred_mean, int_pred_mu, int_pred_logvar)
        y_pred_mean, int_pred_mu, int_pred_logvar = outputs
        return loss_function(target, y_pred_mean, int_pred_mu, int_pred_logvar)
    else:
        # Standard loss function expects prediction, target
        return loss_function(outputs, target)


def _apply_regularization(
    model: ReGNN, regularization: nn.Module, lambda_reg: float
) -> torch.Tensor:
    # This would calculate regularization penalty based on model parameters
    # Example: return lambda_reg * regularization(torch.cat([p.view(-1) for p in model.parameters()]))
    # The actual implementation depends on the 'regularization' module type
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    if hasattr(regularization, "get_penalty"):  # Custom interface
        penalty = regularization.get_penalty(model)
    else:  # Generic, assuming regularization module takes parameters
        # This part is very speculative without knowing the regularization module
        pass  # Placeholder: Actual regularization logic needed
    return lambda_reg * penalty


def _apply_survey_weights(
    loss: torch.Tensor, sample: Dict[str, torch.Tensor]
) -> torch.Tensor:
    if "survey_weights" in sample and sample["survey_weights"] is not None:
        weights = sample["survey_weights"].squeeze()
        if loss.ndim > 0 and weights.ndim > 0 and loss.size() == weights.size():
            return (loss * weights).mean()  # Or .sum() depending on reduction
        else:  # Fallback or error if shapes mismatch
            return loss.mean()  # Or raise error
    return loss.mean()  # Default if no weights or reduction='none' handled by loss


def _collect_probes(
    probes: List[Union[Type[ProbeData], Callable]],
    data_source: str,
    time_value: Union[float, int],
    loss_value: Optional[float] = None,
    model: Optional[ReGNN] = None,
) -> Snapshot:
    """Helper function to collect probe data.

    Args:
        probes: List of probe types or callables
        data_source: Data source identifier
        time_value: Time value for the snapshot
        loss_value: Optional loss value for ObjectiveProbe
        model: Optional model instance for probes that require model access

    Returns:
        Snapshot containing all collected probe data
    """
    snapshot = Snapshot(time=time_value, measurements=[])

    if not probes:
        return snapshot

    for probe in probes:
        if probe == ObjectiveProbe and loss_value is not None:
            # Add loss probe
            snapshot.measurements.append(
                ObjectiveProbe(data_source=data_source, loss=loss_value)
            )
        elif probe == L2NormProbe:
            # Skip L2NormProbe if no model is provided
            if model is None:
                print(f"Skipping {probe.__name__}: model not provided")
                continue
            # Use the compute method to create the L2 norm probe
            l2_probe = L2NormProbe.compute(model, data_source=data_source)
            snapshot.measurements.append(l2_probe)
        elif callable(probe) and probe != ObjectiveProbe:
            # Skip model-dependent probes if no model is provided
            if model is None:
                print(f"Skipping {probe.__name__}: model not provided")
                continue
            # Execute other probe function with model
            try:
                probe_data = probe(model)
                if isinstance(probe_data, ProbeData):
                    snapshot.measurements.append(probe_data)
            except Exception as e:
                # Handle probe execution errors
                print(f"Error executing probe {probe.__name__}: {e}")

    return snapshot


def process_iteration(
    model: ReGNN,
    sample: Dict[str, torch.Tensor],
    loss_function: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights_active: bool = True,
    vae_loss_active: bool = False,
    is_training: bool = True,
    probes: List[Union[Type[ProbeData], Callable]] = [ObjectiveProbe],
    data_source: str = "train",
    epoch: int = -1,
    batch_idx: int = 0,
    dataloader_length: int = 1000,
) -> Snapshot:
    """Run one iteration (batch processing) for training or evaluation.

    Args:
        model: The ReGNN model to process
        sample: Dictionary of tensors for batch processing
        loss_function: Loss function module
        optimizer: Optional optimizer for training
        regularization: Optional regularization module
        lambda_reg: Regularization strength
        survey_weights_active: Whether to use survey weights
        vae_loss_active: Whether VAE loss is active
        is_training: Whether in training mode
        probes: List of probe types or probe functions to collect, defaults to [ObjectiveProbe]
        data_source: Source of data ('train', 'test', 'validate')
        epoch: Current epoch number
        batch_idx: Current batch index
        dataloader_length: Length of the dataloader for normalizing iteration

    Returns:
        Snapshot containing probe data including loss value
    """

    if is_training and optimizer:
        optimizer.zero_grad()

    # Forward pass
    outputs = model(
        sample["moderators"], sample["focal_predictor"], sample["controlled_predictors"]
    )

    # Calculate loss
    loss = _compute_loss(loss_function, outputs, sample["outcome"], vae_loss_active)

    # Apply additive regularization penalty if needed
    if regularization is not None and lambda_reg > 0:
        loss += _apply_regularization(model, regularization, lambda_reg)

    # Apply survey weights if needed and loss reduction is 'none'
    if survey_weights_active and loss.ndim > 0:  # Check if loss is not scalar
        loss = _apply_survey_weights(loss, sample)
    elif loss.ndim > 0:  # If loss is not scalar but no survey weights
        loss = loss.mean()

    if is_training and optimizer:
        loss.backward()
        optimizer.step()

    loss_value = loss.item()

    # Calculate normalized iteration as decimal
    decimal_iteration = epoch + (batch_idx + 1) / dataloader_length

    # Collect probes
    snapshot = _collect_probes(
        probes=probes,
        data_source=data_source,
        time_value=decimal_iteration,
        loss_value=loss_value,
        model=model,
    )

    return snapshot


def process_epoch(
    model: ReGNN,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights_active: bool = True,
    vae_loss_active: bool = False,
    is_training: bool = True,
    probes: List[Union[Type[ProbeData], Callable]] = [ObjectiveProbe],
    epoch: int = -1,
    device: str = "cpu",
) -> Tuple[Snapshot, Trajectory]:
    """Run one epoch of training or evaluation.

    Args:
        model: The ReGNN model to process
        dataloader: DataLoader providing batches
        loss_function: Loss function module
        optimizer: Optional optimizer for training
        regularization: Optional regularization module
        lambda_reg: Regularization strength
        survey_weights_active: Whether to use survey weights
        vae_loss_active: Whether VAE loss is active
        is_training: Whether in training mode
        probes: List of probe types or probe functions to collect, defaults to [ObjectiveProbe]
        epoch: Current epoch number
        device: Device to run on

    Returns:
        Tuple containing:
            - Epoch-level snapshot with aggregated metrics
            - Trajectory containing all batch-level snapshots
    """

    original_return_logvar_state = None
    if hasattr(model, "return_logvar"):
        original_return_logvar_state = model.return_logvar

    if is_training:
        model.train()
    else:
        model.eval()

    if vae_loss_active and hasattr(model, "return_logvar"):
        model.return_logvar = True

    running_loss = 0.0
    num_batches = 0
    data_source = "train" if is_training else "validate"
    batch_trajectory = Trajectory()

    # Get dataloader length for decimal iteration calculation
    dataloader_length = len(dataloader)

    with torch.set_grad_enabled(is_training):  # Manage gradient calculation context
        for batch_idx, sample in enumerate(dataloader):
            # Move sample to device if not already done by DataLoader
            # sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            batch_snapshot = process_iteration(
                model=model,
                sample=sample,
                loss_function=loss_function,
                optimizer=optimizer,
                regularization=regularization,
                lambda_reg=lambda_reg,
                survey_weights_active=survey_weights_active,
                vae_loss_active=vae_loss_active,
                is_training=is_training,
                probes=probes,
                data_source=data_source,
                epoch=epoch,
                batch_idx=batch_idx,
                dataloader_length=dataloader_length,
            )

            # Extract loss from the snapshot
            if batch_snapshot.measurements:
                loss_probes = [
                    m
                    for m in batch_snapshot.measurements
                    if isinstance(m, ObjectiveProbe)
                ]
                if loss_probes:
                    running_loss += loss_probes[0].loss
                    num_batches += 1

            batch_trajectory.append(batch_snapshot)

    # Calculate average loss for the epoch
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Collect epoch-level probes
    epoch_snapshot = _collect_probes(
        probes=probes,
        data_source=data_source,
        time_value=float(epoch),
        loss_value=epoch_loss,
        model=model,
    )

    if hasattr(model, "return_logvar") and original_return_logvar_state is not None:
        model.return_logvar = original_return_logvar_state  # Restore original state

    return epoch_snapshot, batch_trajectory
