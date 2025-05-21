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
from regnn.train.loop import process_epoch  # Use process_epoch

from .base import MacroConfig  # The main configuration object
from .preprocess import preprocess  # For data loading and preprocessing
from .evaluator import (
    get_regression_summary,
    get_thresholded_value,
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
    probe_opts = macro_config.probe  # Added for clarity
    read_cfg = macro_config.read_config  # Added for clarity
    regression_cfg = macro_config.regression  # Added for clarity

    # 1. Preprocessing
    # The preprocess function now takes DataFrameReadInConfig and ModeratedRegressionConfig directly.
    all_dataset = preprocess(read_config=read_cfg, regression_config=regression_cfg)

    # 2. Data Splitting
    # Assuming train_ratio is in training_hp or a dedicated split_config like training_hp.data_split_options.train_ratio
    # For now, using a direct attribute if it exists, else default.
    train_split_ratio = getattr(training_hp, "train_test_split_ratio", 0.8)
    if not (0 < train_split_ratio < 1):
        raise ValueError(
            f"train_test_split_ratio must be between 0 and 1, got {train_split_ratio}"
        )

    train_indices, test_indices = train_test_split(
        num_elems=len(all_dataset.df), train_ratio=train_split_ratio
    )

    train_dataset = all_dataset.get_subset(train_indices)
    test_dataset = all_dataset.get_subset(test_indices)

    # SVD matrix computation remains the same, using train_dataset.config and regnn_cfg
    svd_matrix_for_model = None
    if regnn_cfg.nn_config.svd.enabled:
        if regnn_cfg.nn_config.svd.svd_matrix is not None:
            svd_matrix_for_model = regnn_cfg.nn_config.svd.svd_matrix
        else:  # Compute SVD from train_dataset
            # Ensure moderators in train_dataset.config are correctly resolved if they are lists of lists for ensemble models
            moderator_columns = train_dataset.config.moderators
            if isinstance(
                moderator_columns[0], list
            ):  # Ensemble case, flatten for SVD on combined moderators or handle per model
                # This SVD computation assumes a single SVD matrix. For ensemble models with separate SVDs,
                # this logic would need to be part of the model or a more complex setup.
                # For now, assuming moderators is List[str] for this PCA step.
                # If it's List[List[str]], this will fail. Revisit if ensemble SVD is separate.
                raise NotImplementedError(
                    "SVD computation for ensemble models with separate moderator lists not directly supported here. Pre-compute SVD matrix or ensure moderators is List[str]."
                )

            moderators_np = train_dataset.df[moderator_columns].to_numpy()
            _U, _S, V_computed = torch.pca_lowrank(
                torch.from_numpy(moderators_np).to(torch.float32),
                q=regnn_cfg.nn_config.svd.k_dim,
                center=False,  # As per original logic
                niter=10,  # As per original logic
            )
            V_computed = V_computed.to(torch.float32)
            V_computed.requires_grad = False
            svd_matrix_for_model = V_computed
            regnn_cfg.nn_config.svd.svd_matrix = (
                svd_matrix_for_model  # Save computed SVD matrix back to config
            )

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
    train_torch_dataset = train_dataset.to_torch_dataset(device=training_hp.device)
    dataloader = DataLoader(
        train_torch_dataset,
        batch_size=training_hp.batch_size,
        shuffle=training_hp.shuffle,
    )

    test_dataset_torch: Optional[Dict[str, torch.Tensor]] = None
    test_dataloader: Optional[DataLoader] = None
    if training_hp.get_testset_results and test_dataset:
        test_dataset_torch = test_dataset.to_torch_dataset(device=training_hp.device)
        test_dataloader = DataLoader(
            test_dataset_torch,
            batch_size=training_hp.batch_size,
            shuffle=False,
        )

    # 5. Training Loop
    trajectory_data: List[TrajectoryData] = []
    intermediate_indices: List[np.ndarray] = []

    if eval_opts.evaluate_initial and training_hp.return_trajectory:
        initial_train_moderators = train_dataset.to_tensor(device=training_hp.device)[
            "moderators"
        ]
        initial_train_idx_preds = (
            compute_index_prediction(model, initial_train_moderators).cpu().numpy()
        )
        df_train_with_preds = train_dataset.df_orig.copy()
        df_train_with_preds[eval_opts.index_column_name] = initial_train_idx_preds

        traj_epoch = TrajectoryData(epoch_num_if_applicable=0)
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

        traj_epoch_data = TrajectoryData(
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
