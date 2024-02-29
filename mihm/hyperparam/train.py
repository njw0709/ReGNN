from ..data.trainutils import train_test_val_split
from torch.utils.data import DataLoader
import torch
from ..data.dataset import MIHMDataset
from typing import Sequence, Union
import torch.nn as nn
import torch.optim as optim
from mihm.model.mihm import MIHM
from .eval import evaluate_significance, compute_index_prediction
import pandas as pd

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)


def train(
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
    id: Union[None, str] = None,
):

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
    mseLoss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, sample in enumerate(dataloader):
            optimizer.zero_grad()
            # forward pass
            if model.vae:
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
            # loss = var_adjusted_mse_loss(predicted_epi, label, logvar, lambda_reg=0.1)
            loss = mseLoss(predicted_epi, label)
            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print average loss for epoch
        epoch_loss = running_loss / len(dataloader)
        # evaluation on test set
        loss_test = test(model, test_dataset_torch_sample)
        # evaluate on significance
        if eval:
            assert df_orig is not None
            assert all_interaction_predictors is not None
            index_predictions = compute_index_prediction(
                model, all_interaction_predictors
            )
            interaction_pval, vif = evaluate_significance(
                df_orig, index_predictions, id=id
            )
            print(
                "Epoch {}/{} done!; Training Loss: {}; Testing Loss: {}; Interaction Pval: {}; VIF Heat: {}; VIF Interaction: {};".format(
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    loss_test,
                    interaction_pval,
                    vif[0],
                    vif[1],
                )
            )
        else:
            print(
                "Epoch {}/{} done!; Training Loss: {}; Testing Loss: {};".format(
                    epoch + 1, epochs, epoch_loss, loss_test
                )
            )
    return model


def test(model: MIHM, test_dataset_torch_sample: MIHMDataset):
    model.eval()
    mseLoss = nn.MSELoss()
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


def save_model(model: MIHM, path: str):
    torch.save(model.state_dict(), path)
