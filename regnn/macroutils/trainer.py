from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from regnn.data.dataset import ReGNNDataset
from regnn.data.datautils import train_test_split  # For splitting dataset
from regnn.model.regnn import ReGNN
from regnn.model.base import ReGNNConfig  # Used directly from MacroConfig
from regnn.model.custom_loss import (
    vae_kld_regularized_loss,
    elasticnet_loss,
    lasso_loss,
)
from regnn.eval.eval import compute_index_prediction  # For intermediate index
from regnn.train.base import (
    TrajectoryData,
    TrainingHyperParams,  # Explicitly import for type hint
    LossConfigs,  # Base for loss options
    MSELossConfig,  # Specific MSE config
    KLDLossConfig,  # Specific KLD config
    ElasticNetRegConfig,  # Specific Regularization Config
)
from regnn.train.loop import train_epoch  # Use train_epoch for the main loop

from .base import MacroConfig  # The main configuration object
from .preprocess import preprocess  # For data loading and preprocessing
from .evaluator import (
    get_regression_summary,
    get_thresholded_value,
    test_regnn,  # For loss on test set
)
from .utils import save_regnn  # For saving model

# create_model and setup_loss_and_optimizer will remain here for now
# They are specific to how ReGNN is set up within this macro flow.


def create_model(
    train_dataset_config: Any,  # ReGNNDatasetConfig from all_dataset.config
    training_hyperparams: Any,  # training_hyperparams from macro_config.training
    regnn_model_config: ReGNNConfig,  # regnn_model_config from macro_config.model
) -> ReGNN:
    """Create and initialize the ReGNN model using sub-configs from MacroConfig."""
    # train_dataset_config is an instance of ReGNNDatasetConfig
    moderator_size = len(train_dataset_config.moderators)
    controlled_var_size = len(train_dataset_config.controlled_predictors)

    svd_matrix = None
    if regnn_model_config.nn_config.svd.enabled:
        # SVD is usually performed on the actual data, not just from config.
        # This part might need adjustment if SVD matrix isn't pre-computed and passed in ReGNNConfig
        # Assuming svd_matrix might be passed via ReGNNConfig.nn_config.svd.svd_matrix if precomputed
        # If it needs to be computed here, we need the actual moderator data.
        # For now, let's assume if enabled, svd_matrix is provided in regnn_model_config or needs data.
        # The original trainer did PCA on train_dataset.df. This implies create_model needs data access.
        # This is a slight departure, let's assume svd_matrix is part of regnn_model_config.nn_config.svd.svd_matrix if used
        svd_matrix = regnn_model_config.nn_config.svd.svd_matrix
        # Or, if it MUST be computed here, we'd need train_dataset.df which create_model doesn't have.
        # This indicates a potential need to pass train_dataset to create_model, or precompute SVD matrix
        # and set it in MacroConfig -> ReGNNConfig before calling train.
        # For now, proceeding with assumption it's in regnn_model_config if used.
        pass  # Placeholder for SVD matrix logic if it needs to be computed from data here

    model = ReGNN(
        num_moderators=moderator_size,
        num_controlled=controlled_var_size,
        hidden_layer_sizes=training_hyperparams.hidden_layer_sizes,
        svd=regnn_model_config.nn_config.svd.enabled,
        svd_matrix=svd_matrix,
        k_dim=regnn_model_config.nn_config.svd.k_dim,
        include_bias_focal_predictor=regnn_model_config.include_bias_focal_predictor,
        control_moderators=regnn_model_config.control_moderators,
        batch_norm=regnn_model_config.nn_config.batch_norm,
        vae=regnn_model_config.nn_config.vae,
        output_mu_var=regnn_model_config.nn_config.output_mu_var,
        dropout=regnn_model_config.nn_config.dropout,
        device=training_hyperparams.device,
        n_ensemble=regnn_model_config.nn_config.n_ensemble,
        interaction_direction=regnn_model_config.interaction_direction,
    )

    if training_hyperparams.device == "cuda":
        model.cuda()
    return model


def setup_loss_and_optimizer(
    model: ReGNN,
    training_hyperparams: TrainingHyperParams,  # Use the specific type
    regnn_model_config: ReGNNConfig,
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
                reduction="mean", alpha=reg_config.elastic_net_alpha
            )
        elif reg_config.name == "lasso":
            # Assuming a LassoRegConfig would be similar if it existed formally
            # For now, relies on name and expects regularization_alpha from base RegularizationConfig
            regularization = lasso_loss(reduction="mean")
        # Add other specific regularization types here if needed, e.g.:
        # elif isinstance(reg_config, RidgeRegConfig):
        #     regularization = ridge_loss(reduction="mean")
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
) -> Union[Tuple[ReGNN, List[TrajectoryData]], ReGNN]:
    """
    Train a ReGNN model using the comprehensive MacroConfig.

    Args:
        macro_config: Configuration object containing all necessary parameters.

    Returns:
        Trained model and optionally trajectory data.
    """
    training_hp = macro_config.training
    regnn_cfg = macro_config.model
    eval_opts = macro_config.evaluation
    # probe_opts = macro_config.probe # For while-training probes if used

    # 1. Preprocessing
    # Assuming data_options within eval_opts holds info for preprocess func
    # Adjust if MacroConfig stores these differently (e.g., a dedicated preprocess_params section)
    data_options = (
        eval_opts.data_options
    )  # This is an assumption for where data_path etc. are stored.
    # MacroConfig might have these at a different path.
    # The preprocess function expects: data_path, read_cols, rename_dict, etc.
    # These need to be mapped from data_options or another part of MacroConfig.

    # Example mapping (MUST BE ADJUSTED based on actual MacroConfig.evaluation.data_options structure)
    all_dataset = preprocess(
        data_path=data_options.data_path,
        read_cols=data_options.read_cols,
        rename_dict=data_options.rename_dict,
        binary_cols=data_options.binary_cols,
        categorical_cols=data_options.categorical_cols,
        ordinal_cols=data_options.ordinal_cols,
        continuous_cols=data_options.continuous_cols,
        focal_predictor=data_options.focal_predictor_col,  # from eval_opts
        outcome_col=data_options.outcome_col,  # from eval_opts
        controlled_cols=data_options.controlled_vars_cols,  # from eval_opts
        moderators=data_options.moderator_cols,  # from eval_opts
        survey_weights=(
            data_options.survey_weights_col
            if hasattr(data_options, "survey_weights_col")
            else None
        ),
    )

    # 2. Data Splitting
    # Assuming train_ratio is in training_hp or a dedicated split_config
    train_indices, test_indices = train_test_split(
        num_elems=len(all_dataset.df),
        train_ratio=(
            training_hp.train_test_split_ratio
            if hasattr(training_hp, "train_test_split_ratio")
            else 0.8
        ),
    )

    train_df = all_dataset.df.iloc[train_indices].copy()
    test_df = all_dataset.df.iloc[test_indices].copy()

    # Create ReGNNDataset instances for train and test sets using get_subset
    train_dataset = all_dataset.get_subset(train_indices)
    test_dataset = all_dataset.get_subset(test_indices)

    # If SVD matrix needs to be computed from train_dataset, do it here and update regnn_cfg
    if regnn_cfg.nn_config.svd.enabled and regnn_cfg.nn_config.svd.svd_matrix is None:
        moderators_np = train_dataset.df[train_dataset.config.moderators].to_numpy()
        _U, _S, V_svd = torch.pca_lowrank(
            torch.from_numpy(moderators_np).to(torch.float32),
            q=regnn_cfg.nn_config.svd.k_dim,
            center=False,
            niter=10,
        )
        V_svd = V_svd.to(torch.float32)
        V_svd.requires_grad = False
        # Update the ReGNNConfig in MacroConfig (or a copy)
        # Pydantic models are immutable by default, so care is needed here.
        # Best to create a new SVDConfig and update nn_config, then ReGNNConfig.
        # This is complex with nested Pydantic models. Simpler if ReGNNConfig is mutable for this or svd_matrix is passed directly.
        # For now, assuming regnn_cfg can be updated or ReGNN directly takes the computed matrix.
        # If not, the SVD computation in create_model (which I commented on) needs the actual data.
        # Safest: Pre-populate MacroConfig.model.nn_config.svd.svd_matrix if computed outside.
        # Or, adjust create_model to take train_dataset for SVD computation.
        # Modifying create_model signature:
        # def create_model(train_dataset: ReGNNDataset, training_hyperparams: Any, regnn_model_config: ReGNNConfig)
        # And then pass train_dataset to it.
        # The current create_model takes train_dataset_config, not the dataset itself.
        # This needs reconciliation. For now, I will assume svd_matrix is directly set in macro_config if used.
        # If not, the SVD computation in create_model (which I commented on) needs the actual data.
        # Safest: Pre-populate MacroConfig.model.nn_config.svd.svd_matrix if computed outside.
        # Or, adjust create_model to take train_dataset. Let's adjust create_model for SVD computation.
        pass  # This SVD logic has been moved into the model creation step that takes train_dataset.

    # 3. Model, Loss, Optimizer Setup
    # We need to adjust create_model to take the actual train_dataset for SVD if computed there.
    # Let's refine create_model's SVD part.
    # For now, assuming create_model uses train_dataset.config (moderator_size etc.) and training_hp (device etc.)
    # and regnn_cfg (the rest). If SVD is computed inside create_model, it needs train_dataset.df.
    # For simplicity, let create_model take train_dataset directly.

    # Re-calling create_model with train_dataset instead of just its config
    # And TrainingHyperParams and ReGNNConfig
    model = ReGNN.from_config(
        regnn_cfg
    )  # Use from_config if it handles SVD matrix correctly
    # or call ReGNN directly as before, but SVD matrix handling needs to be clean.
    # The ReGNNConfig now holds all model architectural choices.
    # The TrainingHyperParams has training specific choices (lr, epochs, device etc)

    # If SVD is enabled and matrix is not in config, it MUST be computed from train_dataset.
    # The original `train_regnn` computed SVD from `train_dataset` before initializing `ReGNN`.
    # Let's stick to that pattern: compute SVD here if needed, then pass matrix to ReGNN.
    svd_matrix_for_model = None
    if regnn_cfg.nn_config.svd.enabled:
        if regnn_cfg.nn_config.svd.svd_matrix is not None:
            svd_matrix_for_model = regnn_cfg.nn_config.svd.svd_matrix
        else:  # Compute SVD from train_dataset
            moderators_np = train_dataset.df[train_dataset.config.moderators].to_numpy()
            _U, _S, V_computed = torch.pca_lowrank(
                torch.from_numpy(moderators_np).to(torch.float32),
                q=regnn_cfg.nn_config.svd.k_dim,
                center=False,
                niter=10,
            )
            V_computed = V_computed.to(torch.float32)
            V_computed.requires_grad = False
            svd_matrix_for_model = V_computed
            # Potentially update regnn_cfg if other parts of the system expect svd_matrix to be in the config
            # However, for model creation, we can just pass it.

    model = ReGNN(
        num_moderators=len(train_dataset.config.moderators),
        num_controlled=len(train_dataset.config.controlled_predictors),
        hidden_layer_sizes=training_hp.hidden_layer_sizes,
        svd=regnn_cfg.nn_config.svd.enabled,
        svd_matrix=svd_matrix_for_model,
        k_dim=regnn_cfg.nn_config.svd.k_dim,
        include_bias_focal_predictor=regnn_cfg.include_bias_focal_predictor,
        control_moderators=regnn_cfg.control_moderators,
        batch_norm=regnn_cfg.nn_config.batch_norm,
        vae=regnn_cfg.nn_config.vae,
        output_mu_var=regnn_cfg.nn_config.output_mu_var,
        dropout=regnn_cfg.nn_config.dropout,
        device=training_hp.device,
        n_ensemble=regnn_cfg.nn_config.n_ensemble,
        interaction_direction=regnn_cfg.interaction_direction,
    )
    if training_hp.device == "cuda":
        model.cuda()

    loss_func, regularization, optimizer = setup_loss_and_optimizer(
        model, training_hp, regnn_cfg
    )

    # 4. DataLoader
    train_torch_dataset = train_dataset.to_torch_dataset(device=training_hp.device)
    dataloader = DataLoader(
        train_torch_dataset,
        batch_size=training_hp.batch_size,
        shuffle=training_hp.shuffle,
    )

    test_dataset_torch: Optional[Dict[str, torch.Tensor]] = None
    if training_hp.get_testset_results and test_dataset:
        test_dataset_torch = test_dataset.to_tensor(device=training_hp.device)

    # 5. Training Loop
    trajectory_data: List[TrajectoryData] = []
    intermediate_indices: List[np.ndarray] = []

    # Initial evaluation (if configured)
    if eval_opts.evaluate_initial and training_hp.return_trajectory:
        # For initial evaluation, a df with predictions is needed for get_regression_summary
        # We can compute index predictions for train and test datasets here
        initial_train_moderators = train_dataset.to_tensor(device=training_hp.device)[
            "moderators"
        ]
        initial_train_idx_preds = (
            compute_index_prediction(model, initial_train_moderators).cpu().numpy()
        )
        df_train_with_preds = train_dataset.df_orig.copy()
        df_train_with_preds[eval_opts.index_column_name] = initial_train_idx_preds

        traj_epoch = TrajectoryData(epoch_num_if_applicable=0)  # Indicate initial eval
        traj_epoch.regression_summary = get_regression_summary(
            model, train_dataset, df_train_with_preds, training_hp, regnn_cfg, eval_opts
        )

        if training_hp.get_testset_results and test_dataset:
            initial_test_moderators = test_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            initial_test_idx_preds = (
                compute_index_prediction(model, initial_test_moderators).cpu().numpy()
            )
            df_test_with_preds = test_dataset.df_orig.copy()
            df_test_with_preds[eval_opts.index_column_name] = initial_test_idx_preds
            traj_epoch.regression_summary_test = get_regression_summary(
                model,
                test_dataset,
                df_test_with_preds,
                training_hp,
                regnn_cfg,
                eval_opts,
            )
        trajectory_data.append(traj_epoch)

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
        train_epoch_vae_loss_flag = (
            regnn_cfg.nn_config.vae
            and regnn_cfg.nn_config.output_mu_var
            and is_kld_loss
        )

        epoch_loss, l2_lengths = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_function=loss_func,
            regularization=regularization,
            lambda_reg=current_lambda_reg,
            survey_weights=use_survey_weights,
            vae_loss=train_epoch_vae_loss_flag,
            get_l2_lengths=(
                macro_config.probe.get_l2_lengths
                if hasattr(macro_config.probe, "get_l2_lengths")
                else False
            ),
        )

        current_test_loss: Optional[float] = None
        if training_hp.get_testset_results and test_dataset_torch:
            current_test_loss = test_regnn(
                model,
                test_dataset_torch,
                use_survey_weights,
                (
                    (
                        training_hp.loss_options.regularization is not None
                        and isinstance(
                            training_hp.loss_options.regularization,
                            (ElasticNetRegConfig),
                        )
                    )
                    if training_hp.loss_options.regularization
                    else False
                ),
                regularization,
            )

        if macro_config.probe.save_intermediate_index:
            # Use all_dataset for consistent intermediate index calculation
            all_moderators = all_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            idx_pred = compute_index_prediction(model, all_moderators)
            intermediate_indices.append(idx_pred.cpu().numpy())

        if (
            macro_config.probe.save_model_epochs
            and epoch % macro_config.probe.save_model_epochs == 0
        ):
            save_regnn(
                model,
                data_id=(
                    f"{eval_opts.data_options.file_id_prefix}_{epoch}"
                    if hasattr(eval_opts.data_options, "file_id_prefix")
                    else str(epoch)
                ),
            )

        traj_epoch_data = TrajectoryData(
            train_loss=epoch_loss,
            test_loss=current_test_loss if current_test_loss is not None else -1.0,
            l2=l2_lengths,
            epoch_num_if_applicable=epoch + 1,
        )

        if eval_opts.evaluate_epochs and epoch % eval_opts.evaluate_epochs == 0:
            quietly = (
                (epoch % (eval_opts.evaluate_epochs * 3) != 0)
                if hasattr(eval_opts, "quiet_eval_multiplier")
                else (epoch % 30 != 0)
            )

            # For train set evaluation
            train_moderators_eval = train_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            train_idx_preds_eval = (
                compute_index_prediction(model, train_moderators_eval).cpu().numpy()
            )
            df_train_eval_with_preds = train_dataset.df_orig.copy()
            df_train_eval_with_preds[eval_opts.index_column_name] = train_idx_preds_eval
            threshold_val_train = get_thresholded_value(
                model, train_dataset
            )  # Use train_dataset for thresholding stats

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
                threshold_val_test = get_thresholded_value(
                    model, test_dataset
                )  # Use test_dataset for thresholding stats

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

            # Early stopping (ensure criteria are available in TrainingHyperParams or ProbeOptions)
            if training_hp.stopping_options and training_hp.stopping_options.early_stop:
                stop_opts = training_hp.stopping_options
                p_val_train = reg_summary_train.get("interaction term p value", 1.0)
                p_val_test = (
                    reg_summary_test.get("interaction term p value", 1.0)
                    if reg_summary_test
                    else 1.0
                )

                if (
                    p_val_train < stop_opts.early_stop_criterion
                    and (
                        p_val_test < stop_opts.early_stop_criterion
                        if training_hp.get_testset_results and test_dataset
                        else True
                    )
                    and epoch > stop_opts.stop_after
                ):
                    print(f"Reached early stopping criterion at epoch: {epoch}")
                    if training_hp.return_trajectory:
                        trajectory_data.append(traj_epoch_data)
                    break

        if training_hp.return_trajectory:
            trajectory_data.append(traj_epoch_data)

        # Print progress
        printout = (
            f"Epoch {epoch + 1}/{training_hp.epochs} | Train Loss: {epoch_loss:.6f}"
        )
        if current_test_loss is not None:
            printout += f" | Test Loss: {current_test_loss:.6f}"
        if eval_opts.evaluate_epochs and epoch % eval_opts.evaluate_epochs == 0:
            if traj_epoch_data.regression_summary:
                printout += f" | Train P-val: {traj_epoch_data.regression_summary.get('interaction term p value'):.4f}"
            if traj_epoch_data.regression_summary_test:
                printout += f" | Test P-val: {traj_epoch_data.regression_summary_test.get('interaction term p value'):.4f}"
        print(printout)

    # Final evaluation (if configured)
    if eval_opts.evaluate_final:
        final_moderators = all_dataset.to_tensor(device=training_hp.device)[
            "moderators"
        ]
        final_idx_preds = (
            compute_index_prediction(model, final_moderators).cpu().numpy()
        )
        df_final_with_preds = (
            all_dataset.df_orig.copy()
        )  # Use all_dataset.df_orig for final eval
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

    if macro_config.probe.save_intermediate_index and intermediate_indices:
        intermediate_indices_np = np.hstack(intermediate_indices)
        df_indices = pd.DataFrame(intermediate_indices_np)
        indices_path = (
            macro_config.probe.intermediate_indices_file_path
            if hasattr(macro_config.probe, "intermediate_indices_file_path")
            else "indices.dta"
        )
        df_indices.to_stata(indices_path)
        print(f"Intermediate indices saved to {indices_path}")

    if training_hp.return_trajectory:
        return model, trajectory_data
        return model
