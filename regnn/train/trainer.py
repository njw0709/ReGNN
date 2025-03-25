import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regnn.model.regnn import ReGNN
from regnn.data.dataset import ReGNNDataset, TorchReGNNDataset
from regnn.model.custom_loss import (
    vae_kld_regularized_loss,
    elasticnet_loss,
    lasso_loss,
)
from typing import Dict, List, Any, Optional, Tuple, Union, Sequence
import numpy as np
import pandas as pd
from .utils import get_l2_length, save_regnn
from .base import TrainingConfig, TrajectoryData
from .evaluation import eval_regnn, test_regnn
from .loop import train_iteration
from .eval import compute_index_prediction


def train_regnn(
    all_dataset: ReGNNDataset,
    train_dataset: ReGNNDataset,
    config: TrainingConfig,
    test_dataset: Optional[ReGNNDataset] = None,
    df_orig: Optional[pd.DataFrame] = None,
) -> Union[Tuple[ReGNN, List[TrajectoryData]], ReGNN]:
    """
    Train a ReGNN model with the given datasets and configuration.

    Args:
        all_dataset: Complete dataset for final evaluation
        train_dataset: Dataset for training
        config: Training configuration
        test_dataset: Optional dataset for testing
        df_orig: Original dataframe for evaluation

    Returns:
        Trained model and optionally trajectory data
    """
    # Initialize trajectory data if needed
    trajectory_data = []

    # Create torch dataset for training
    train_torch_dataset = train_dataset.to_torch_dataset(device=config.device)

    # Create test tensor dataset if needed
    if test_dataset is not None and config.get_testset_results:
        test_dataset_torch = test_dataset.to_tensor(device=config.device)

    # Create dataloader
    dataloader = DataLoader(
        train_torch_dataset, batch_size=config.batch_size, shuffle=config.shuffle
    )

    # Get dimensions for model initialization
    moderator_size = len(train_dataset.config.moderators)
    controlled_var_size = len(train_dataset.config.controlled_predictors)

    # Initialize model with SVD if needed
    if config.svd:
        # Dimension reduction using PCA
        moderators_np = train_dataset.df[train_dataset.config.moderators].to_numpy()
        U, S, V = torch.pca_lowrank(
            torch.from_numpy(moderators_np), q=config.k_dims, center=False, niter=10
        )
        V = V.to(torch.float32)
        V.requires_grad = False

        model = ReGNN(
            moderator_size,
            controlled_var_size,
            config.hidden_layer_sizes,
            svd=config.svd,
            svd_matrix=V,
            k_dim=config.k_dims,
            include_bias_focal_predictor=config.include_bias_focal_predictor,
            control_moderators=True,
            batch_norm=True,
            vae=config.vae,
            output_mu_var=config.vae_loss,
            dropout=config.dropout,
            device=config.device,
            n_ensemble=config.n_models,
            interaction_direction=config.interaction_direction,
        )
    else:
        model = ReGNN(
            moderator_size,
            controlled_var_size,
            config.hidden_layer_sizes,
            svd=config.svd,
            include_bias_focal_predictor=config.include_bias_focal_predictor,
            control_moderators=True,
            batch_norm=True,
            vae=config.vae,
            output_mu_var=config.vae_loss,
            dropout=config.dropout,
            device=config.device,
            n_ensemble=config.n_models,
            interaction_direction=config.interaction_direction,
        )

    # Move model to device
    if config.device == "cuda":
        model.cuda()

    # Setup loss function
    if config.vae_loss:
        if config.survey_weights:
            loss_func = vae_kld_regularized_loss(
                lambda_reg=config.vae_lambda, reduction="none"
            )
        else:
            loss_func = vae_kld_regularized_loss(
                lambda_reg=config.vae_lambda, reduction="mean"
            )
    else:
        if config.survey_weights:
            loss_func = nn.MSELoss(reduction="none")
        else:
            loss_func = nn.MSELoss()

    # Setup regularization if needed
    if config.elasticnet:
        regularization = elasticnet_loss(reduction="mean", alpha=0.005)
    elif config.lasso:
        regularization = lasso_loss(reduction="mean")
    else:
        regularization = None

    # Setup optimizer
    optimizer = optim.AdamW(
        [
            {
                "params": model.index_prediction_model.parameters(),
                "weight_decay": config.weight_decay_nn,
            },
            {"params": model.mmr_parameters},
        ],
        lr=config.lr,
        weight_decay=config.weight_decay_regression,
    )

    # Initialize intermediate indices if needed
    if config.save_intermediate_index:
        intermediate_indices = []

    # Initial evaluation if needed
    if config.evaluate and config.return_trajectory:
        traj_epoch = TrajectoryData()

        regression_summary = eval_regnn(
            model,
            train_dataset,
            df_orig,
            config.regress_cmd,
            use_stata=config.use_stata,
            file_id=config.file_id,
            interaction_direction=config.interaction_direction,
        )
        traj_epoch.regression_summary = regression_summary

        if config.get_testset_results and test_dataset is not None:
            traj_epoch.regression_summary_test = eval_regnn(
                model,
                test_dataset,
                df_orig,
                config.regress_cmd,
                use_stata=config.use_stata,
                file_id=config.file_id,
                interaction_direction=config.interaction_direction,
            )

        trajectory_data.append(traj_epoch)

    # Main training loop
    for epoch in range(config.epochs):
        # Train one epoch
        epoch_loss, l2_lengths = train_iteration(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_function=loss_func,
            regularization=regularization,
            lambda_reg=config.lambda_reg,
            survey_weights=config.survey_weights,
            vae_loss=config.vae_loss,
            get_l2_lengths=config.get_l2_lengths,
        )

        # Test on test dataset if needed
        if config.get_testset_results and test_dataset is not None:
            loss_test = test_regnn(
                model,
                test_dataset_torch,
                survey_weights=config.survey_weights,
                regularize=(config.elasticnet or config.lasso),
                regularization=regularization,
            )

        # Save intermediate index if needed
        if config.save_intermediate_index:
            moderators = all_dataset.to_tensor(device=config.device)["moderators"]
            index_prediction = compute_index_prediction(model, moderators)
            intermediate_indices.append(index_prediction)

        # Save model if needed
        if config.save_model and epoch % 10 == 0:
            if config.file_id is not None:
                save_regnn(model, data_id=f"{config.file_id}_{epoch}")
            else:
                save_regnn(model, data_id=f"{epoch}")

        # Create trajectory data if needed
        if config.return_trajectory:
            traj_epoch = TrajectoryData(
                train_loss=epoch_loss,
                test_loss=(
                    loss_test
                    if config.get_testset_results and test_dataset is not None
                    else -1
                ),
                l2=l2_lengths if config.get_l2_lengths else None,
            )

        # Evaluate on significance if needed
        if config.evaluate and epoch % config.eval_epoch == 0:
            quietly = epoch % 30 != 0

            if model.include_bias_focal_predictor:
                thresholded_value = (
                    model.interactor_bias.cpu().detach().numpy().item(0)
                    * all_dataset.mean_std_dict[all_dataset.config.focal_predictor][1]
                    + all_dataset.mean_std_dict[all_dataset.config.focal_predictor][0]
                )
            else:
                thresholded_value = 0.0

            regression_summary = eval_regnn(
                model,
                train_dataset,
                df_orig,
                config.regress_cmd,
                use_stata=config.use_stata,
                file_id=config.file_id,
                threshold=model.include_bias_focal_predictor,
                thresholded_value=thresholded_value,
                interaction_direction=config.interaction_direction,
            )

            if config.get_testset_results and test_dataset is not None:
                regression_summary_test = eval_regnn(
                    model,
                    test_dataset,
                    df_orig,
                    config.regress_cmd,
                    use_stata=config.use_stata,
                    file_id=config.file_id,
                    quietly=quietly,
                    threshold=model.include_bias_focal_predictor,
                    thresholded_value=thresholded_value,
                    interaction_direction=config.interaction_direction,
                )

            if config.return_trajectory:
                traj_epoch.regression_summary = regression_summary
                if config.get_testset_results and test_dataset is not None:
                    traj_epoch.regression_summary_test = regression_summary_test

            # Early stopping if needed
            if (
                config.early_stop
                and regression_summary["interaction term p value"]
                < config.early_stop_criterion
                and regression_summary_test["interaction term p value"]
                < config.early_stop_criterion
                and epoch > config.stop_after
            ):
                print(f"Reached early stopping criterion at epoch: {epoch}")
                break

        # Add trajectory data
        if config.return_trajectory:
            trajectory_data.append(traj_epoch)

        # Print progress
        printout = (
            f"Epoch {epoch + 1}/{config.epochs} done! Training Loss: {epoch_loss:.6f}"
        )
        if config.get_testset_results and test_dataset is not None:
            printout += f" Testing Loss: {loss_test:.6f}"
        if config.evaluate and epoch % config.eval_epoch == 0:
            printout += f" Regression Summary: {regression_summary}"
            if config.get_testset_results and test_dataset is not None:
                printout += f" Testset Regression Summary: {regression_summary_test}"
        print(printout)

    # Final evaluation
    if config.evaluate:
        if model.include_bias_focal_predictor:
            thresholded_value = (
                model.interactor_bias.cpu().detach().numpy().item(0)
                * all_dataset.mean_std_dict[all_dataset.config.focal_predictor][1]
                + all_dataset.mean_std_dict[all_dataset.config.focal_predictor][0]
            )
        else:
            thresholded_value = 0.0

        final_summary = eval_regnn(
            model,
            all_dataset,
            df_orig,
            config.regress_cmd,
            use_stata=config.use_stata,
            file_id=f"{config.file_id}_final" if config.file_id else "final",
            quietly=False,
            threshold=model.include_bias_focal_predictor,
            thresholded_value=thresholded_value,
            interaction_direction=config.interaction_direction,
        )
        print(f"Final evaluation: {final_summary}")

    # Save intermediate indices if needed
    if config.save_intermediate_index:
        intermediate_indices = np.hstack(intermediate_indices)
        df = pd.DataFrame(intermediate_indices)
        df.to_stata("indices.dta")

    # Return model and trajectory data if needed
    if config.return_trajectory:
        return model, trajectory_data
    else:
        return model
