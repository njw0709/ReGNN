import torch
from regnn.model.regnn import ReGNN
from typing import Optional, Union, Tuple, List, Dict, Any
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
from .base import DataFrameReadInConfig, ModeratedRegressionConfig

# Imports for the new function
from regnn.probe import ProbeData, ObjectiveProbe, OLSModeratedResultsProbe, L2NormProbe
from regnn.probe.dataclass.probe_config import DataSource


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
    optimizer_opts = training_hyperparams.optimizer_config
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
    nn_params = [p for p in model.index_prediction_model.parameters()]
    if model.include_bias_focal_predictor:
        nn_params.append(model.xf_bias)
    optimizer = optim.AdamW(
        [
            {
                "params": nn_params,
                "weight_decay": optimizer_opts.weight_decay.weight_decay_nn,
                "lr": optimizer_opts.lr.lr_nn,
            },
            {
                "params": model.mmr_parameters,
                "weight_decay": optimizer_opts.weight_decay.weight_decay_regression,
                "lr": optimizer_opts.lr.lr_regression,
            },
        ],
        weight_decay=0.0,  # Top-level weight_decay is 0 as it's handled per param group
    )
    return loss_func, regularization, optimizer


def generate_stata_command(
    data_readin_config: DataFrameReadInConfig,
    regression_config: ModeratedRegressionConfig,
) -> str:
    """
    Generate a Stata regression command based on the configuration.
    Uses the rename_dict from data_readin_config to map internal column names to Stata variable names.

    Returns:
        str: The complete Stata regression command
    """

    # Helper function to get the correct variable name
    def get_var_name(col: str) -> str:
        # If col is a value in rename_dict, return its key
        # If col is a key in rename_dict, return the original name
        # Otherwise return the original name
        rename_dict = data_readin_config.rename_dict
        if col in rename_dict.values():
            # Find the key for this value
            for key, value in rename_dict.items():
                if value == col:
                    return key
        return col

    # Start with basic regression command
    cmd_parts = ["regress"]

    # Add outcome and focal predictor with interaction
    outcome = get_var_name(regression_config.outcome_col)
    focal = get_var_name(regression_config.focal_predictor)

    # Add outcome and focal predictor with interaction
    cmd_parts.append(outcome)

    # Add appropriate prefix to focal predictor based on its type
    if (
        data_readin_config.binary_cols
        and regression_config.focal_predictor in data_readin_config.binary_cols
    ) or (
        data_readin_config.categorical_cols
        and regression_config.focal_predictor in data_readin_config.categorical_cols
    ):
        cmd_parts.append(f"i.{focal}")
    else:
        cmd_parts.append(f"c.{focal}")

    # Add summary index and focal predictor interactions
    cmd_parts.append(f"c.{focal}#c.{regression_config.index_column_name}")

    # Add linear terms
    linear_terms = regression_config.controlled_cols
    if regression_config.control_moderators:
        linear_terms += regression_config.moderators
    for col in linear_terms:
        col_name = get_var_name(col)
        # Check if the column is binary or categorical in the data config
        if data_readin_config.binary_cols and col in data_readin_config.binary_cols:
            cmd_parts.append(f"i.{col_name}")
        elif (
            data_readin_config.categorical_cols
            and col in data_readin_config.categorical_cols
        ):
            cmd_parts.append(f"i.{col_name}")
        elif data_readin_config.ordinal_cols and col in data_readin_config.ordinal_cols:
            cmd_parts.append(f"i.{col_name}")
        else:
            assert col in data_readin_config.continuous_cols
            cmd_parts.append(f"c.{col_name}")

    # Add survey weights if specified
    if data_readin_config.survey_weight_col is not None:
        weight_col = get_var_name(data_readin_config.survey_weight_col)
        cmd_parts.append(f"[pweight={weight_col}]")

    return " ".join(cmd_parts)


def format_epoch_printout(
    base_printout: str, epoch_probe_results: List[ProbeData]
) -> str:
    """
    Formats and appends relevant information from epoch-level probe results to a base printout string.

    Args:
        base_printout: The initial printout string (e.g., epoch number and training loss).
        epoch_probe_results: A list of ProbeData objects from the current epoch's probes.

    Returns:
        An augmented string with information from recognized probes.
    """
    additional_info = []

    # Sort by data_source then probe_type for consistent ordering if multiple results exist
    # This is a simple sort, more complex might be needed if specific order is critical
    # For now, primarily to group by data_source for display
    # sorted_results = sorted(epoch_probe_results, key=lambda r: (r.data_source, r.probe_type_name))

    # It might be better to collect specific types of results first, then format them
    # e.g., all test objectives, then all test p-values

    test_objective_str: Optional[str] = None
    val_objective_str: Optional[str] = None  # If you have validation data source
    train_l2_norm_str: Optional[str] = None
    test_l2_norm_str: Optional[str] = None
    train_pval_str: Optional[str] = None
    test_pval_str: Optional[str] = None

    for result in epoch_probe_results:
        if isinstance(result, ObjectiveProbe):
            # Assuming objective_name is like "total_loss", "accuracy", etc.
            # And result.objective is the float value.
            # And result.data_source is "TRAIN", "TEST", etc.
            ds_str = result.data_source.upper()
            obj_name = result.objective_name or "Objective"
            obj_val_str = (
                f"{result.objective:.4f}"
                if isinstance(result.objective, float)
                else str(result.objective)
            )

            if ds_str == DataSource.TEST.value:
                test_objective_str = f"Test {obj_name}: {obj_val_str}"
            elif ds_str == DataSource.VALIDATION.value:
                val_objective_str = f"Val {obj_name}: {obj_val_str}"
            # Train objective is already in base_printout, typically

        elif isinstance(result, OLSModeratedResultsProbe):
            ds_str = result.data_source.upper()
            pval_str = (
                f"{result.interaction_pval:.4f}"
                if result.interaction_pval is not None
                else "N/A"
            )
            r2_str = f"{result.rsquared:.3f}" if result.rsquared is not None else "N/A"

            current_pval_info = f"{ds_str} PVal: {pval_str} (R2: {r2_str})"
            if ds_str == DataSource.TRAIN.value:
                train_pval_str = current_pval_info
            elif ds_str == DataSource.TEST.value:
                test_pval_str = current_pval_info

        elif isinstance(result, L2NormProbe):
            ds_str = (
                result.data_source.upper()
            )  # L2NormProbe might not always have a data_source if global
            # For now, assume it might, or handle if it's None/ALL
            main_norm_str = (
                f"{result.main_norm:.4e}"
                if isinstance(result.main_norm, float)
                else str(result.main_norm)
            )
            index_norm_str = (
                f"{result.index_norm:.4e}"
                if isinstance(result.index_norm, float)
                else str(result.index_norm)
            )
            l2_info = f"{ds_str} L2: Main={main_norm_str}, Index={index_norm_str}"
            if (
                ds_str == DataSource.TRAIN.value
            ):  # Or if data_source is None/ALL for a global L2
                train_l2_norm_str = l2_info
            elif (
                ds_str == DataSource.TEST.value
            ):  # L2 on test data might not be common, but possible if model changes
                test_l2_norm_str = l2_info
            elif (
                result.data_source.upper() == DataSource.ALL.value
            ):  # Handle global L2 Norm
                train_l2_norm_str = (
                    f"Global L2: Main={main_norm_str}, Index={index_norm_str}"
                )

    # Assemble the printout string in a preferred order
    if test_objective_str:
        additional_info.append(test_objective_str)
    if val_objective_str:
        additional_info.append(val_objective_str)
    if train_pval_str:
        additional_info.append(train_pval_str)
    if test_pval_str:
        additional_info.append(test_pval_str)
    if (
        train_l2_norm_str and DataSource.ALL.value not in train_l2_norm_str
    ):  # Avoid double print if global was captured
        additional_info.append(train_l2_norm_str)
    elif train_l2_norm_str:  # If it's the global one
        additional_info.insert(0, train_l2_norm_str)  # Put global L2 first

    if test_l2_norm_str:
        additional_info.append(test_l2_norm_str)

    if additional_info:
        return base_printout + " | " + " | ".join(additional_info)
    return base_printout
