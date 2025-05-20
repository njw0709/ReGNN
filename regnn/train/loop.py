import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regnn.model.regnn import ReGNN
from typing import Dict, Optional


def _forward_pass(model, sample, vae_loss):
    if model.vae and vae_loss:
        return model(
            sample["moderators"],
            sample["focal_predictor"],
            sample["controlled_predictors"],
        )
    return model(
        sample["moderators"],
        sample["focal_predictor"],
        sample["controlled_predictors"],
    )


def _compute_loss(loss_function, outputs, label, vae_loss):
    label = label.unsqueeze(1)
    if vae_loss:
        predicted_epi, mu, logvar = outputs
        return loss_function(predicted_epi, label, mu, logvar)
    else:
        predicted_epi = outputs
        return loss_function(predicted_epi, label)


def _apply_regularization(model, regularization, lambda_reg):
    return lambda_reg * sum(
        regularization(p) for p in model.index_prediction_model.parameters()
    )


def _apply_survey_weights(loss, sample):
    if not hasattr(loss, "shape"):
        raise ValueError(
            "Survey weights require per-sample losses. "
            "Ensure the loss function is configured with reduction='none'."
        )
    if loss.shape[0] != sample["weights"].shape[0]:
        raise ValueError(
            "Mismatch between loss batch size and sample weights batch size."
        )
    return (loss * sample["weights"]).mean()


def train_iteration(
    model: ReGNN,
    sample: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights: bool = True,
    vae_loss: bool = False,
) -> float:
    """Run one training iteration on a single batch."""

    optimizer.zero_grad()

    # Forward pass
    outputs = _forward_pass(model, sample, vae_loss)

    # Calculate loss
    loss = _compute_loss(loss_function, outputs, sample["outcome"], vae_loss)

    # Apply regularization if needed
    if regularization is not None:
        loss += _apply_regularization(model, regularization, lambda_reg)

    # Apply survey weights if needed
    if survey_weights:
        loss = _apply_survey_weights(loss, sample)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(
    model: ReGNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights: bool = True,
    vae_loss: bool = False,
) -> float:
    """Run one training iteration (epoch)"""
    model.train()
    if vae_loss:
        model.return_logvar = True

    running_loss = 0.0

    for batch_idx, sample in enumerate(dataloader):
        batch_loss = train_iteration(
            model=model,
            sample=sample,
            optimizer=optimizer,
            loss_function=loss_function,
            regularization=regularization,
            lambda_reg=lambda_reg,
            survey_weights=survey_weights,
            vae_loss=vae_loss,
        )
        running_loss += batch_loss

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader)

    return epoch_loss
