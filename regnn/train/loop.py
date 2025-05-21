import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regnn.model.regnn import ReGNN
from typing import Dict, Optional, Tuple, Any

from regnn.probe.fns.regnn import get_l2_length


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


def process_iteration(
    model: ReGNN,
    sample: Dict[str, torch.Tensor],
    loss_function: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights_active: bool = True,  # Renamed for clarity
    vae_loss_active: bool = False,  # Renamed for clarity
    is_training: bool = True,
) -> float:
    """Run one iteration (batch processing) for training or evaluation."""

    if is_training and optimizer:
        optimizer.zero_grad()

    # Forward pass
    # Model's forward pass output (single tensor or tuple) is handled by _compute_loss based on vae_loss_active.
    outputs = model(
        sample["moderators"], sample["focal_predictor"], sample["controlled_predictors"]
    )

    # Calculate loss
    loss = _compute_loss(loss_function, outputs, sample["outcome"], vae_loss_active)

    # Apply additive regularization penalty if needed
    if regularization is not None and lambda_reg > 0:
        # Note: Weight decay is handled by optimizer. This is for additive penalties.
        loss += _apply_regularization(model, regularization, lambda_reg)

    # Apply survey weights if needed and loss reduction is 'none'
    # (MSELoss with reduction='none' returns per-element losses)
    if survey_weights_active and loss.ndim > 0:  # Check if loss is not scalar
        # and loss_function has reduction='none' (implicit assumption)
        loss = _apply_survey_weights(loss, sample)
    elif (
        loss.ndim > 0
    ):  # If loss is not scalar (e.g. reduction='none') but no survey weights
        loss = loss.mean()

    if is_training and optimizer:
        loss.backward()
        optimizer.step()

    return loss.item()


def process_epoch(
    model: ReGNN,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights_active: bool = True,  # Renamed
    vae_loss_active: bool = False,  # Renamed
    is_training: bool = True,
    get_l2_lengths: bool = False,
    device: str = "cpu",  # Added device for potential model.compute_l2_lengths
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Run one epoch of training or evaluation."""

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

    with torch.set_grad_enabled(is_training):  # Manage gradient calculation context
        for batch_idx, sample in enumerate(dataloader):
            # Move sample to device if not already done by DataLoader
            # sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            batch_loss = process_iteration(
                model=model,
                sample=sample,
                loss_function=loss_function,
                optimizer=optimizer,
                regularization=regularization,
                lambda_reg=lambda_reg,
                survey_weights_active=survey_weights_active,
                vae_loss_active=vae_loss_active,
                is_training=is_training,
            )
            running_loss += batch_loss
            num_batches += 1

    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0

    l2_lengths_data: Optional[Dict[str, Any]] = None
    if get_l2_lengths:
        l2_lengths_data = get_l2_length(model)

    if hasattr(model, "return_logvar") and original_return_logvar_state is not None:
        model.return_logvar = original_return_logvar_state  # Restore original state

    return epoch_loss, l2_lengths_data
