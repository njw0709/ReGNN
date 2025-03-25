import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regnn.model.regnn import ReGNN

from typing import Dict, List, Optional, Tuple
from .utils import get_l2_length
from .config import TrajectoryData


def train_iteration(
    model: ReGNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    regularization: Optional[nn.Module] = None,
    lambda_reg: float = 0.1,
    survey_weights: bool = True,
    vae_loss: bool = False,
    get_l2_lengths: bool = False,
) -> Tuple[float, Optional[List[Dict[str, float]]]]:
    """Run one training iteration (epoch)"""
    model.train()
    if vae_loss:
        model.return_logvar = True

    running_loss = 0.0
    l2_lengths = []

    for batch_idx, sample in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        if model.vae:
            if vae_loss:
                predicted_epi, mu, logvar = model(
                    sample["moderators"],
                    sample["focal_predictor"],
                    sample["controlled_predictors"],
                )
            else:
                predicted_epi = model(
                    sample["moderators"],
                    sample["focal_predictor"],
                    sample["controlled_predictors"],
                )
        else:
            predicted_epi = model(
                sample["moderators"],
                sample["focal_predictor"],
                sample["controlled_predictors"],
            )

        label = torch.unsqueeze(sample["outcome"], 1)

        # Calculate loss
        if vae_loss:
            loss = loss_function(predicted_epi, label, mu, logvar)
        else:
            loss = loss_function(predicted_epi, label)

        # Add regularization if needed
        if regularization is not None:
            regloss = lambda_reg * sum(
                regularization(p) for p in model.index_prediction_model.parameters()
            )
            loss += regloss

        # Apply survey weights if needed
        if survey_weights:
            try:
                # For loss with reduction='none', shape is available
                assert loss.shape[0] == sample["weights"].shape[0]
                loss = (loss * sample["weights"]).mean()
            except (AttributeError, IndexError):
                # For loss with reduction='mean', shape is not available
                # In this case, we can't apply weights directly to the reduced loss
                # Log a warning or handle as appropriate for your use case
                pass

        # Backward pass and optimization
        loss.backward()

        # Calculate L2 lengths if needed
        if get_l2_lengths:
            l2 = get_l2_length(model)
            l2_lengths.append(l2)

        optimizer.step()
        running_loss += loss.item()

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader)

    return epoch_loss, l2_lengths if get_l2_lengths else None
