import torch
from regnn.model.regnn import ReGNN
from typing import Optional, Union, Tuple
import os
from regnn.constants import TEMP_DIR
import numpy as np
import torch.nn as nn
import torch.optim as optim
from regnn.model import (
    vae_kld_regularized_loss,
    elasticnet_loss,
    lasso_loss,
)
from regnn.train import (
    TrainingHyperParams,
    MSELossConfig,
    KLDLossConfig,
    ElasticNetRegConfig,
)


def save_model(
    model: torch.nn.Module,
    model_type: str = "model",
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Optional[str] = None,
) -> str:
    """Save PyTorch model to disk

    Args:
        model: PyTorch model to save
        model_type: Type of model for filename prefix (e.g. 'regnn', 'mlp')
        save_dir: Directory to save model in
        data_id: Optional identifier to include in filename

    Returns:
        str: Path to saved model file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate filename
    if data_id is not None:
        model_name = os.path.join(save_dir, f"{model_type}_{data_id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"{model_type}_{num_files}.pt")

    # Save model
    torch.save(model.state_dict(), model_name)
    return model_name


def load_model(
    model: torch.nn.Module,
    model_path: str,
    map_location: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """Load PyTorch model from disk

    Args:
        model: Instantiated PyTorch model to load weights into
        model_path: Path to saved model file
        map_location: Optional device to map model to (e.g. 'cpu', 'cuda')

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    return model


def compute_index_prediction(
    model: ReGNN, interaction_predictors: torch.Tensor
) -> np.ndarray:

    index_model = model.index_prediction_model
    index_model.to(interaction_predictors.device).eval()
    if index_model.vae:
        index_prediction, log_var = index_model(interaction_predictors)
    else:
        index_prediction = index_model(interaction_predictors)
    index_prediction = index_prediction.detach().cpu().numpy()

    return index_prediction


def compute_svd(moderators_np: np.ndarray, k_dim=int) -> torch.Tensor:
    _U, _S, V_computed = torch.pca_lowrank(
        torch.from_numpy(moderators_np).to(torch.float32),
        q=k_dim,
        center=False,  # As per original logic
        niter=10,  # As per original logic
    )
    V_computed = V_computed.to(torch.float32)
    V_computed.requires_grad = False
    return V_computed


def setup_loss_and_optimizer(
    model: ReGNN,
    training_hyperparams: TrainingHyperParams,  # Use the specific type
) -> Tuple[nn.Module, Optional[nn.Module], optim.Optimizer]:
    """Setup loss function, regularization, and optimizer based on TrainingHyperParams and ReGNNConfig."""

    loss_opts = training_hyperparams.loss_options
    loss_func: nn.Module
    regularization: Optional[nn.Module] = None

    # 1. Setup Loss Function
    # Validation for KLDLossConfig compatibility with regnn_model_config (vae=True, output_mu_var=True)
    # is handled by MacroConfig's model_validator.
    if isinstance(loss_opts, KLDLossConfig):
        loss_func = vae_kld_regularized_loss(
            lambda_reg=loss_opts.lamba_reg,
            reduction=loss_opts.reduction,
        )
    elif isinstance(loss_opts, MSELossConfig):
        loss_func = nn.MSELoss(reduction=loss_opts.reduction)
    else:
        # This case should ideally be prevented by Pydantic if using discriminated unions for LossConfigs subtypes.
        # If LossConfigs is a Union of MSELossConfig | KLDLossConfig, this path might not be reachable
        # if the input `loss_options` is always one of the specific types.
        raise ValueError(
            f"Unsupported loss configuration type: {type(loss_opts)}. "
            f"Expected MSELossConfig or KLDLossConfig. Ensure training_hyperparams.loss_options is correctly initialized."
        )

    # 2. Setup Regularization
    if loss_opts.regularization:
        reg_config = loss_opts.regularization
        if isinstance(reg_config, ElasticNetRegConfig):
            regularization = elasticnet_loss(
                reduction=loss_opts.reduction, alpha=reg_config.elastic_net_alpha
            )
        elif reg_config.name == "lasso":
            # Assuming a LassoRegConfig would be similar if it existed formally
            # For now, relies on name and expects regularization_alpha from base RegularizationConfig
            regularization = lasso_loss(reduction=loss_opts.reduction)
        else:
            print(
                f"Warning: Unknown or non-specific regularization type specified: {reg_config.name}. No additive penalty will be applied beyond optimizer weight decay."
            )

    # 3. Setup Optimizer
    weight_decay_conf = None
    # Check if loss_opts has a 'weight_decay' attribute, which MSELossConfig and KLDLossConfig do.
    if hasattr(loss_opts, "weight_decay") and loss_opts.weight_decay is not None:
        weight_decay_conf = loss_opts.weight_decay

    wd_nn = 0.0
    wd_reg = 0.0
    if weight_decay_conf:  # Ensure it's not None
        wd_nn = (
            weight_decay_conf.weight_decay_nn
            if weight_decay_conf.weight_decay_nn is not None
            else 0.0
        )
        wd_reg = (
            weight_decay_conf.weight_decay_regression
            if weight_decay_conf.weight_decay_regression is not None
            else 0.0
        )

    optimizer = optim.AdamW(
        [
            {
                "params": model.index_prediction_model.parameters(),
                "weight_decay": wd_nn,
            },
            {"params": model.mmr_parameters, "weight_decay": wd_reg},
        ],
        lr=training_hyperparams.lr,
        weight_decay=0.0,  # Top-level weight_decay is 0 as it's handled per param group
    )
    return loss_func, regularization, optimizer
