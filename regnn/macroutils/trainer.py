from typing import List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from regnn.data import train_test_split
from regnn.model import ReGNN
from regnn.train import KLDLossConfig, process_epoch
from regnn.probe import (
    L2NormProbe,
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
    ObjectiveProbe,
    Trajectory,
    Snapshot,
)

from .base import MacroConfig  # The main configuration object
from .preprocess import preprocess  # For data loading and preprocessing
from .evaluator import (
    regression_eval_regnn,
)
from .utils import compute_index_prediction, save_model  # For saving model
from .utils import (
    compute_svd,
    setup_loss_and_optimizer,
)  # Added import for compute_svd function


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
    probe_opts = macro_config.probe
    regression_eval_opts = probe_opts.regression_eval_opts

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
    train_trajectory_data = Trajectory()
    test_trajectory_data: Optional[Trajectory] = None
    if probe_opts.get_testset_results:
        test_trajectory_data = Trajectory()
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

        # Define probes based on configuration
        probes = [ObjectiveProbe]
        if probe_opts.get_l2_lengths:
            probes.append(L2NormProbe)

        # Process training epoch
        train_epoch_snapshot, train_batch_trajectory = process_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_function=loss_func,
            regularization=regularization,
            lambda_reg=current_lambda_reg,
            survey_weights_active=use_survey_weights,
            vae_loss_active=train_epoch_vae_loss_active,
            is_training=True,
            probes=probes,
            epoch=epoch,
            device=training_hp.device,
        )

        # Extract epoch loss from snapshot
        objective_probe = train_epoch_snapshot.get(ObjectiveProbe)
        epoch_loss = objective_probe.loss if objective_probe else 0.0
        printout = (
            f"Epoch {epoch + 1}/{training_hp.epochs} | Train Loss: {epoch_loss:.6f}"
        )

        # Process test dataset
        current_test_loss: Optional[float] = None
        test_epoch_snapshot: Optional[Snapshot] = None
        test_batch_trajectory: Optional[Trajectory] = None

        if training_hp.get_testset_results and test_dataloader:

            test_epoch_snapshot, test_batch_trajectory = process_epoch(
                model=model,
                dataloader=test_dataloader,
                loss_function=loss_func,
                optimizer=None,
                regularization=regularization,
                lambda_reg=current_lambda_reg,
                survey_weights_active=use_survey_weights,
                vae_loss_active=train_epoch_vae_loss_active,
                is_training=False,
                probes=probes,
                epoch=epoch,
                device=training_hp.device,
            )

            objective_probe = test_epoch_snapshot.get(ObjectiveProbe)
            current_test_loss = objective_probe.loss if objective_probe else 0.0
            printout += f" | Test Loss: {current_test_loss:.6f}"

        # post train epoch operations
        if probe_opts.save_intermediate_index:
            all_moderators = all_dataset.to_tensor(device=training_hp.device)[
                "moderators"
            ]
            idx_pred = compute_index_prediction(model, all_moderators)
            intermediate_indices.append(idx_pred.cpu().numpy())

        if probe_opts.save_model and epoch % probe_opts.save_model_epochs == 0:
            model_file_prefix = (
                f"{probe_opts.model_save_name}-{probe_opts.file_id}"
                if probe_opts.file_id
                else probe_opts.model_save_name
            )

            save_model(
                model,
                model_type="regnn",
                save_dir=probe_opts.save_dir,
                data_id=f"{model_file_prefix}_{epoch}",
            )

        # Update trajectory data
        if probe_opts.return_trajectory:
            # Add batch-level trajectories
            train_trajectory_data.extend(train_batch_trajectory)
            if test_batch_trajectory:
                test_trajectory_data.extend(test_batch_trajectory)
            # Add epoch-level snapshots
            train_trajectory_data.append(train_epoch_snapshot)
            if test_epoch_snapshot:
                test_trajectory_data.append(test_epoch_snapshot)

        # run regression for evaluation
        if (
            regression_eval_opts.evaluate
            and epoch % regression_eval_opts.eval_epochs == 0
        ):
            # Evaluate on training data
            train_ols_results, train_vif_results = regression_eval_regnn(
                model=model,
                eval_regnn_dataset=train_dataset,
                eval_options=regression_eval_opts,
                device=training_hp.device,
                data_source="train",
            )
            train_p_val = train_ols_results.interaction_term_p_value
            printout += f" | Train P-val: {train_p_val:.4f}"

            # Add regression results to train trajectory
            if probe_opts.return_trajectory:
                train_epoch_snapshot.add(OLSModeratedResultsProbe, train_ols_results)
                train_epoch_snapshot.add(
                    VarianceInflationFactorProbe, train_vif_results
                )

            # Evaluate on test data if available
            test_ols_results = None
            test_vif_results = None
            if test_dataset:
                test_ols_results, test_vif_results = regression_eval_regnn(
                    model=model,
                    eval_regnn_dataset=test_dataset,
                    eval_options=regression_eval_opts,
                    device=training_hp.device,
                    data_source="test",
                )
                test_p_val = test_ols_results.interaction_term_p_value
                printout += f" | Test P-val: {test_p_val:.4f}"

                # Add regression results to test trajectory
                if probe_opts.return_trajectory and test_epoch_snapshot:
                    test_epoch_snapshot.add(OLSModeratedResultsProbe, test_ols_results)
                    test_epoch_snapshot.add(
                        VarianceInflationFactorProbe, test_vif_results
                    )

            # Early stopping check
            if training_hp.stopping_options and training_hp.stopping_options.enabled:
                stop_opts = training_hp.stopping_options
                p_val_train = train_p_val
                p_val_test = (
                    test_ols_results.interaction_term_p_value
                    if test_ols_results
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
                    break

        print(printout)

    if regression_eval_opts.post_training_eval:
        # Final evaluation on all data
        final_ols_results, final_vif_results = regression_eval_regnn(
            model=model,
            eval_regnn_dataset=all_dataset,
            eval_options=regression_eval_opts,
            device=training_hp.device,
            data_source="all",
        )
        final_summary = {
            "interaction term p value": final_ols_results.interaction_term_p_value,
            "interaction term coefficient": final_ols_results.interaction_term_coefficient,
            "vif": final_vif_results.vif_value,
        }
        print(f"Final evaluation summary: {final_summary}")

    if probe_opts.save_intermediate_index and intermediate_indices:
        intermediate_indices_np = np.hstack(intermediate_indices)
        df_indices = pd.DataFrame(intermediate_indices_np)
        indices_path = os.path.join(probe_opts.save_dir, "indices.dta")
        df_indices.to_stata(indices_path)
        print(f"Intermediate indices saved to {indices_path}")

    if probe_opts.return_trajectory:
        return model, train_trajectory_data, test_trajectory_data
    return model
