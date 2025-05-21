from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from regnn.data.datautils import train_test_split  # For splitting dataset
from regnn.model.regnn import ReGNN
from regnn.model.custom_loss import (
    vae_kld_regularized_loss,
    elasticnet_loss,
    lasso_loss,
)
from regnn.eval.eval import compute_index_prediction  # For intermediate index
from regnn.train.base import (
    TrainingHyperParams,  # Explicitly import for type hint
    MSELossConfig,  # Specific MSE config
    KLDLossConfig,  # Specific KLD config
    ElasticNetRegConfig,  # Specific Regularization Config
)
from regnn.probe import Trajectory
from regnn.train.loop import process_epoch  # Use process_epoch

from .base import MacroConfig  # The main configuration object
from .preprocess import preprocess  # For data loading and preprocessing
from .evaluator import (
    get_regression_summary,
    get_thresholded_value,
)
from .utils import save_regnn  # For saving model
from .utils import compute_svd  # Added import for compute_svd function


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


def train(
    macro_config: MacroConfig,
) -> Union[Tuple[ReGNN, List[Trajectory]], ReGNN]:
    """
    Train a ReGNN model using the comprehensive MacroConfig.

    Args:
        macro_config: Configuration object containing all necessary parameters.

    Returns:
        Trained model and optionally trajectory data.
    """
    read_cfg = macro_config.read_config
    regression_cfg = macro_config.regression

    regnn_cfg = macro_config.model
    training_hp = macro_config.training
    eval_opts = macro_config.evaluation
    probe_opts = macro_config.probe

    # 1. Preprocessing
    # The preprocess function now takes DataFrameReadInConfig and ModeratedRegressionConfig directly.
    all_dataset = preprocess(read_config=read_cfg, regression_config=regression_cfg)

    # 2. Data Splitting
    train_indices, test_indices = train_test_split(
        num_elems=len(all_dataset),
        train_ratio=training_hp.train_test_split_ratio,
        seed=training_hp.train_test_split_seed,
    )

    train_dataset = all_dataset.get_subset(train_indices)
    test_dataset = all_dataset.get_subset(test_indices)

    # SVD matrix computation remains the same, using train_dataset.config and regnn_cfg
    if regnn_cfg.nn_config.svd.enabled:
        # Compute SVD from train_dataset
        moderator_columns = train_dataset.config.moderators
        if isinstance(moderator_columns[0], list):  # Ensemble case
            # Compute SVD for each list of moderators separately
            svd_matrices = []
            for mod_list in moderator_columns:
                moderators_np = train_dataset.df[mod_list].to_numpy()
                svd_matrix = compute_svd(
                    moderators_np, k_dim=regnn_cfg.nn_config.svd.k_dim
                )
                svd_matrices.append(svd_matrix)
        else:  # Single list case
            moderators_np = train_dataset.df[moderator_columns].to_numpy()
            svd_matrices = compute_svd(
                moderators_np, k_dim=regnn_cfg.nn_config.svd.k_dim
            )

        regnn_cfg.nn_config.svd.svd_matrix = svd_matrices

    # 3. Model Initialization using from_config
    # ReGNNConfig (regnn_cfg.nn_config.num_moderators) is assumed to be the source of truth for num_moderators,
    # validated upstream (e.g., in MacroConfig) for consistency with n_ensemble and dataset moderator structure.
    model = ReGNN.from_config(
        model_config=regnn_cfg,
    )
    if training_hp.device == "cuda":
        model.cuda()

    loss_func, regularization, optimizer = setup_loss_and_optimizer(
        model, training_hp, regnn_cfg  # regnn_cfg is macro_config.model
    )

    # 4. DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_hp.batch_size,
        shuffle=training_hp.shuffle,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_hp.batch_size,
        shuffle=False,
    )

    # 5. Training Loop
    trajectory_data: List[Trajectory] = []
    intermediate_indices: List[np.ndarray] = []

    for epoch in range(training_hp.epochs):
        current_lambda_reg = 0.0
        if training_hp.loss_options.regularization:
            current_lambda_reg = (
                training_hp.loss_options.regularization.regularization_alpha
            )

        use_survey_weights = (
            training_hp.use_survey_weights
            if hasattr(training_hp, "use_survey_weights")
            else False
        )

        is_kld_loss = isinstance(training_hp.loss_options, KLDLossConfig)
        train_epoch_vae_loss_active = (
            regnn_cfg.nn_config.vae
            and regnn_cfg.nn_config.output_mu_var
            and is_kld_loss
        )

        epoch_loss, l2_lengths = process_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_function=loss_func,
            regularization=regularization,
            lambda_reg=current_lambda_reg,
            survey_weights_active=use_survey_weights,
            vae_loss_active=train_epoch_vae_loss_active,
            is_training=True,
            get_l2_lengths=(
                probe_opts.get_l2_lengths
                if hasattr(probe_opts, "get_l2_lengths")
                else False
            ),
            device=training_hp.device,
        )

        current_test_loss: Optional[float] = None
        if training_hp.get_testset_results and test_dataloader:
            test_epoch_vae_loss_active = train_epoch_vae_loss_active

            current_test_loss, _ = process_epoch(
                model=model,
                dataloader=test_dataloader,
                loss_function=loss_func,
                optimizer=None,
                regularization=regularization,
                lambda_reg=current_lambda_reg,
                survey_weights_active=use_survey_weights,
                vae_loss_active=test_epoch_vae_loss_active,
                is_training=False,
                get_l2_lengths=False,
                device=training_hp.device,
            )

        if probe_opts.save_intermediate_index:  # Use probe_opts
            all_moderators = all_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            idx_pred = compute_index_prediction(model, all_moderators)
            intermediate_indices.append(idx_pred.cpu().numpy())

        if (
            probe_opts.save_model_epochs
            and epoch % probe_opts.save_model_epochs == 0
            and probe_opts.save_model_epochs > 0
        ):
            model_file_prefix = (
                probe_opts.file_id if probe_opts.file_id else probe_opts.model_save_name
            )
            save_regnn(model, data_id=f"{model_file_prefix}_{epoch}")

        traj_epoch_data = Trajectory(
            train_loss=epoch_loss,
            test_loss=current_test_loss if current_test_loss is not None else -1.0,
            l2=l2_lengths,
            epoch_num_if_applicable=epoch + 1,
        )

        if (
            eval_opts.evaluate_epochs
            and epoch % eval_opts.evaluate_epochs == 0
            and eval_opts.evaluate_epochs > 0
        ):
            quietly = (
                (
                    epoch
                    % (eval_opts.evaluate_epochs * eval_opts.quiet_eval_multiplier)
                    != 0
                )
                if hasattr(eval_opts, "quiet_eval_multiplier")
                and eval_opts.quiet_eval_multiplier > 0
                else (epoch % 30 != 0)
            )

            train_moderators_eval = train_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            train_idx_preds_eval = (
                compute_index_prediction(model, train_moderators_eval).cpu().numpy()
            )
            df_train_eval_with_preds = train_dataset.df_orig.copy()
            df_train_eval_with_preds[eval_opts.index_column_name] = train_idx_preds_eval
            threshold_val_train = get_thresholded_value(model, train_dataset)

            reg_summary_train = get_regression_summary(
                model,
                train_dataset,
                df_train_eval_with_preds,
                training_hp,
                regnn_cfg,
                eval_opts,
                threshold_val_train,
                quietly,
            )
            traj_epoch_data.regression_summary = reg_summary_train

            reg_summary_test = None
            if training_hp.get_testset_results and test_dataset:
                test_moderators_eval = test_dataset.to_tensor(
                    device=training_hp.device
                )["moderators"]
                test_idx_preds_eval = (
                    compute_index_prediction(model, test_moderators_eval).cpu().numpy()
                )
                df_test_eval_with_preds = test_dataset.df_orig.copy()
                df_test_eval_with_preds[eval_opts.index_column_name] = (
                    test_idx_preds_eval
                )
                threshold_val_test = get_thresholded_value(model, test_dataset)

                reg_summary_test = get_regression_summary(
                    model,
                    test_dataset,
                    df_test_eval_with_preds,
                    training_hp,
                    regnn_cfg,
                    eval_opts,
                    threshold_val_test,
                    quietly,
                )
                traj_epoch_data.regression_summary_test = reg_summary_test

            if training_hp.stopping_options and training_hp.stopping_options.enabled:
                stop_opts = training_hp.stopping_options
                p_val_train = (
                    reg_summary_train.get("interaction term p value", 1.0)
                    if reg_summary_train
                    else 1.0
                )
                p_val_test = (
                    reg_summary_test.get("interaction term p value", 1.0)
                    if reg_summary_test
                    else 1.0
                )

                if (
                    p_val_train < stop_opts.criterion
                    and (
                        p_val_test < stop_opts.criterion
                        if training_hp.get_testset_results and test_dataset
                        else True
                    )
                    and epoch > stop_opts.patience
                ):
                    print(f"Reached early stopping criterion at epoch: {epoch}")
                    if training_hp.return_trajectory:
                        trajectory_data.append(traj_epoch_data)
                    break

        if training_hp.return_trajectory:
            trajectory_data.append(traj_epoch_data)

        printout = (
            f"Epoch {epoch + 1}/{training_hp.epochs} | Train Loss: {epoch_loss:.6f}"
        )
        if current_test_loss is not None:
            printout += f" | Test Loss: {current_test_loss:.6f}"
        if (
            eval_opts.evaluate_epochs
            and epoch % eval_opts.evaluate_epochs == 0
            and eval_opts.evaluate_epochs > 0
        ):
            if traj_epoch_data.regression_summary:
                printout += f" | Train P-val: {traj_epoch_data.regression_summary.get('interaction term p value'):.4f}"
            if traj_epoch_data.regression_summary_test:
                printout += f" | Test P-val: {traj_epoch_data.regression_summary_test.get('interaction term p value'):.4f}"
        print(printout)

    if eval_opts.evaluate_final:
        final_moderators = all_dataset.to_tensor(device=training_hp.device)[
            "moderators"
        ]
        final_idx_preds = (
            compute_index_prediction(model, final_moderators).cpu().numpy()
        )
        df_final_with_preds = all_dataset.df_orig.copy()
        df_final_with_preds[eval_opts.index_column_name] = final_idx_preds
        final_threshold_val = get_thresholded_value(model, all_dataset)

        final_summary = get_regression_summary(
            model,
            all_dataset,
            df_final_with_preds,
            training_hp,
            regnn_cfg,
            eval_opts,
            final_threshold_val,
            quietly=False,
        )
        print(f"Final evaluation summary: {final_summary}")

    if probe_opts.save_intermediate_index and intermediate_indices:
        intermediate_indices_np = np.hstack(intermediate_indices)
        df_indices = pd.DataFrame(intermediate_indices_np)
        indices_path = (
            probe_opts.intermediate_indices_file_path
            if hasattr(probe_opts, "intermediate_indices_file_path")
            else "indices.dta"
        )
        df_indices.to_stata(indices_path)
        print(f"Intermediate indices saved to {indices_path}")

    if training_hp.return_trajectory:
        return model, trajectory_data
    return model
