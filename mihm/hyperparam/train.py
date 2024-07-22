from ..data.trainutils import train_test_val_split
from torch.utils.data import DataLoader
import torch
from ..data.dataset import MIHMDataset
from typing import Sequence, Union
import torch.nn as nn
import torch.optim as optim
from mihm.model.mihm import MIHM
from mihm.model.custom_loss import var_adjusted_mse_loss, vae_kld_regularized_loss
from .eval import (
    evaluate_significance,
    compute_index_prediction,
    evaluate_significance_stata,
)
from .constants import TEMP_DIR
import pandas as pd
import os
from ray import train
from ray.train import Checkpoint
import pickle
import tempfile

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def train_mihm(
    train_mihm_dataset: MIHMDataset,
    test_mihm_dataset: MIHMDataset,
    hidden_layer_sizes: Sequence[int],
    vae: bool,
    svd: bool,
    k_dims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str = TRAIN_DEVICE,
    shuffle: bool = True,
    eval: bool = False,
    df_orig: Union[None, pd.DataFrame] = None,
    all_interaction_predictors: Union[None, torch.Tensor] = None,
    file_id: Union[None, str] = None,
    save_model: bool = False,
    ray_tune: bool = False,
    use_stata: bool = False,
    return_trajectory: bool = False,
    vae_loss: bool = False,
    vae_lambda: float = 0.1,
    dropout: float = 0.1,
):

    if return_trajectory:
        trajectory_data = []
    outcome_var = train_mihm_dataset.outcome_original_name
    # create dataset
    train_dataset = train_mihm_dataset.to_torch_dataset(device=device)
    test_dataset_torch_sample = test_mihm_dataset.to_tensor(device=device)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    interaction_var_size = train_mihm_dataset.interaction_predictors.__len__()
    controlled_var_size = train_mihm_dataset.controlled_predictors.__len__()
    if svd:
        # dim reduction using PCA
        interaction_vars_np = train_mihm_dataset.df[
            train_mihm_dataset.interaction_predictors
        ].to_numpy()
        U, S, V = torch.pca_lowrank(
            torch.from_numpy(interaction_vars_np), q=k_dims, center=False, niter=10
        )
        V = V.to(torch.float32)
        V.requires_grad = False
        model = MIHM(
            interaction_var_size,
            controlled_var_size,
            hidden_layer_sizes,
            svd=svd,
            svd_matrix=V,
            k_dim=k_dims,
            include_interactor_bias=True,
            concatenate_interaction_vars=True,
            batch_norm=True,
            vae=vae,
            dropout=dropout,
            device=device,
        )
    else:
        model = MIHM(
            interaction_var_size,
            controlled_var_size,
            hidden_layer_sizes,
            svd=svd,
            include_interactor_bias=True,
            concatenate_interaction_vars=True,
            batch_norm=True,
            vae=vae,
            device=device,
        )

    if device == "cuda":
        model.cuda()

    # setup loss and optimizer
    if vae_loss:
        lossFunc = vae_kld_regularized_loss(lambda_reg=vae_lambda)
    else:
        lossFunc = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # checkpoint
    if ray_tune:
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint.pt")
                )
                start_epoch = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["mihm_state_dict"])
                optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    if return_trajectory:
        traj_epoch = {}
        traj_epoch["train_loss"] = -1
        traj_epoch["test_loss"] = -1
        # initial evaluation
        if eval:
            interaction_pval = eval_mihm(
                model, test_mihm_dataset, df_orig, use_stata=use_stata, file_id=file_id
            )
            traj_epoch["interaction_pval"] = interaction_pval
        trajectory_data.append(traj_epoch)

    for epoch in range(start_epoch, epochs):
        model.train()
        if vae_loss:
            model.return_logvar = True
        running_loss = 0.0
        for batch_idx, sample in enumerate(dataloader):
            optimizer.zero_grad()
            # forward pass
            if model.vae:
                if vae_loss:
                    predicted_epi, mu, logvar = model(
                        sample["interaction_predictors"],
                        sample["interactor"],
                        sample["controlled_predictors"],
                    )
                else:
                    predicted_epi = model(
                        sample["interaction_predictors"],
                        sample["interactor"],
                        sample["controlled_predictors"],
                    )
            else:
                predicted_epi = model(
                    sample["interaction_predictors"],
                    sample["interactor"],
                    sample["controlled_predictors"],
                )
            label = torch.unsqueeze(sample["outcome"], 1)
            if vae_loss:
                loss = lossFunc(predicted_epi, label, mu, logvar)
            else:
                loss = lossFunc(predicted_epi, label)
            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print average loss for epoch
        epoch_loss = running_loss / len(dataloader)
        # evaluation on test set
        loss_test = test_mihm(model, test_dataset_torch_sample)
        if return_trajectory:
            traj_epoch = {}
            traj_epoch["train_loss"] = epoch_loss
            traj_epoch["test_loss"] = loss_test
        if ray_tune:
            report_dict = {"mean_MSE": epoch_loss, "test_MSE": loss_test}

        # save model
        if save_model:
            if epoch % 10 == 0:
                if file_id is not None:
                    save_mihm(model, id="{}_{}".format(file_id, epoch))
                else:
                    save_mihm(model, id="{}".format(epoch))

        # evaluate on significance
        if eval:
            interaction_pval = eval_mihm(
                model, test_mihm_dataset, df_orig, use_stata=use_stata, file_id=file_id
            )
            if ray_tune:
                report_dict["interaction_pval"] = interaction_pval
                report_dict["composite_metric"] = interaction_pval + loss_test
            if return_trajectory:
                traj_epoch["interaction_pval"] = interaction_pval
                # traj_epoch["index_predictions"] = index_predictions
            print(
                "Epoch {}/{} done!; Training Loss: {}; Testing Loss: {}; Interaction Pval: {};".format(
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    loss_test,
                    interaction_pval,
                )
            )
        else:
            print(
                "Epoch {}/{} done!; Training Loss: {}; Testing Loss: {};".format(
                    epoch + 1, epochs, epoch_loss, loss_test
                )
            )
        if ray_tune:
            checkpoint_data = {
                "epoch": epoch,
                "mihm_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(checkpoint_data, os.path.join(tmpdir, "checkpoint.pt"))
                train.report(report_dict, checkpoint=Checkpoint.from_directory(tmpdir))
        if return_trajectory:
            trajectory_data.append(traj_epoch)

    if return_trajectory:
        return model, trajectory_data
    else:
        return model


def eval_mihm(
    model: MIHM,
    test_mihm_dataset: MIHMDataset,
    df_orig,
    use_stata: bool = True,
    file_id: Union[str, None] = None,
):
    test_interaction_predictors = test_mihm_dataset.to_tensor(device="cuda")[
        "interaction_predictors"
    ]
    outcome_var = test_mihm_dataset.outcome_original_name
    index_predictions = compute_index_prediction(model, test_interaction_predictors)
    try:
        if use_stata:
            interaction_pval, _ = evaluate_significance_stata(
                df_orig.iloc[test_mihm_dataset.df.index].copy(),
                outcome_var,
                index_predictions,
                id=file_id,
            )
        else:
            interaction_pval = evaluate_significance(
                df_orig.iloc[test_mihm_dataset.df.index].copy(),
                outcome_var,
                index_predictions,
            )
    except Exception as e:
        print(e)
        interaction_pval = 0.1
    return interaction_pval


def test_mihm(model: MIHM, test_dataset_torch_sample: MIHMDataset):
    model.eval()
    mseLoss = nn.MSELoss()
    model.return_logvar = False
    with torch.no_grad():
        predicted_epi = model(
            test_dataset_torch_sample["interaction_predictors"],
            test_dataset_torch_sample["interactor"],
            test_dataset_torch_sample["controlled_predictors"],
        )
        loss_test = mseLoss(
            predicted_epi, torch.unsqueeze(test_dataset_torch_sample["outcome"], 1)
        )

    return loss_test.item()


def save_mihm(
    model: MIHM,
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    id: Union[str, None] = None,
):
    if id is not None:
        model_name = os.path.join(save_dir, f"mihm_model_{id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"mihm_model_{num_files}.pt")
    torch.save(model.state_dict(), model_name)
