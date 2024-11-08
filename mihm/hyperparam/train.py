from torch.utils.data import DataLoader
import torch
from ..data.dataset import MIHMDataset
from typing import Sequence, Union
import torch.nn as nn
import torch.optim as optim
from mihm.model.mihm import MIHM
from mihm.model.custom_loss import vae_kld_regularized_loss, elasticnet_loss, lasso_loss
from .eval import (
    evaluate_significance,
    compute_index_prediction,
    evaluate_significance_stata,
)
from .constants import TEMP_DIR
import pandas as pd
import os
import traceback

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DEVICE = "cpu"
torch.manual_seed(0)


def get_gradient_norms(model: MIHM):
    grad_norms = {}
    main_parameters = [model.focal_predictor_main_weight, model.predicted_index_weight]
    main_parameters += [p for p in model.controlled_var_weights.parameters()]
    grad_main = [p.grad.norm(2).item() for p in main_parameters]
    index_model_params = [p for p in model.index_prediction_model.parameters()]
    grad_index = [p.grad.norm(2).item() for p in index_model_params]
    grad_norms["main"] = grad_main
    grad_norms["index"] = grad_index
    return grad_norms


def get_l2_length(model: MIHM):
    l2_lengths = {}
    main_parameters = [model.focal_predictor_main_weight, model.predicted_index_weight]
    main_parameters += [p for p in model.controlled_var_weights.parameters()][:-1]
    main_parameters = torch.cat(main_parameters, dim=1)
    main_param_l2 = main_parameters.norm(2).item()

    index_norm = model.predicted_index_weight.norm(2).item()
    l2_lengths["main"] = main_param_l2
    l2_lengths["index"] = index_norm
    return l2_lengths


def train_mihm(
    all_heat_dataset: MIHMDataset,
    train_mihm_dataset: MIHMDataset,
    hidden_layer_sizes: Sequence[int],
    vae: bool,
    svd: bool,
    k_dims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay_regression: float,
    weight_decay_nn: float,
    regress_cmd: str,
    test_mihm_dataset: Union[MIHMDataset, None] = None,
    device: str = TRAIN_DEVICE,
    shuffle: bool = True,
    evaluate: bool = False,
    eval_epoch: int = 10,
    get_testset_results: bool = True,
    df_orig: Union[None, pd.DataFrame] = None,
    file_id: Union[None, str] = None,
    save_model: bool = False,
    use_stata: bool = False,
    return_trajectory: bool = False,
    vae_loss: bool = False,
    vae_lambda: float = 0.1,
    dropout: float = 0.1,
    n_models: int = 1,
    elasticnet: bool = False,
    lasso: bool = False,
    lambda_reg: float = 0.1,
    survey_weights: bool = True,
    include_bias_focal_predictor: bool = True,
    interaction_direction: str = "positive",
    get_l2_lengths: bool = True,
    early_stop: bool = True,
    early_stop_criterion: float = 0.01,
    stop_after: int = 100,
):

    if return_trajectory:
        trajectory_data = []
    # create dataset
    train_dataset = train_mihm_dataset.to_torch_dataset(device=device)

    if test_mihm_dataset is not None:
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
            include_bias_focal_predictor=include_bias_focal_predictor,
            control_moderators=True,
            batch_norm=True,
            vae=vae,
            output_mu_var=vae_loss,
            dropout=dropout,
            device=device,
            n_ensemble=n_models,
            interation_direction=interaction_direction,
        )
    else:
        model = MIHM(
            interaction_var_size,
            controlled_var_size,
            hidden_layer_sizes,
            svd=svd,
            include_bias_focal_predictor=include_bias_focal_predictor,
            control_moderators=True,
            batch_norm=True,
            vae=vae,
            output_mu_var=vae_loss,
            dropout=dropout,
            device=device,
            n_ensemble=n_models,
            interation_direction=interaction_direction,
        )

    if device == "cuda":
        model.cuda()

    # setup loss and optimizer
    if vae_loss:
        if survey_weights:
            lossFunc = vae_kld_regularized_loss(lambda_reg=vae_lambda, reduction="none")
        else:
            lossFunc = vae_kld_regularized_loss(lambda_reg=vae_lambda, reduction="mean")
    else:
        if survey_weights:
            lossFunc = nn.MSELoss(reduction="none")
        else:
            lossFunc = nn.MSELoss()
    if elasticnet:
        regularization = elasticnet_loss(reduction="mean", alpha=0.005)
    if lasso:
        regularization = lasso_loss(reduction="mean")
    if not elasticnet and not lasso:
        regularization = None
    optimizer = optim.AdamW(
        [
            {
                "params": model.index_prediction_model.parameters(),
                "weight_decay": weight_decay_nn,
            },
            {"params": model.mmr_parameters},
        ],
        lr=lr,
        weight_decay=weight_decay_regression,
    )
    # learning rate scheduler
    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    # checkpoint
    start_epoch = 0

    if return_trajectory:
        traj_epoch = {}
        traj_epoch["train_loss"] = -1
        traj_epoch["test_loss"] = -1
        # initial evaluation
        if evaluate:
            regression_summary = eval_mihm(
                model,
                train_mihm_dataset,
                df_orig,
                regress_cmd,
                use_stata=use_stata,
                file_id=file_id,
                interaction_direction=interaction_direction,
            )
            traj_epoch["regression_summary"] = regression_summary
            if get_testset_results:
                assert test_mihm_dataset is not None
                traj_epoch["regression_summary_test"] = eval_mihm(
                    model,
                    test_mihm_dataset,
                    df_orig,
                    regress_cmd,
                    use_stata=use_stata,
                    file_id=file_id,
                    interaction_direction=interaction_direction,
                )
        trajectory_data.append(traj_epoch)

    for epoch in range(start_epoch, epochs):
        model.train()
        l2_lengths = []
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
            if elasticnet or lasso:
                regloss = lambda_reg * sum(
                    regularization(p) for p in model.index_prediction_model.parameters()
                )
                loss += regloss
            # backward pass
            if survey_weights:
                assert loss.shape[0] == sample["weights"].shape[0]
                loss = (loss * sample["weights"]).mean()
            loss.backward()
            if get_l2_lengths:
                l2 = get_l2_length(model)
                l2_lengths.append(l2)
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()

        # print average loss for epoch
        epoch_loss = running_loss / len(dataloader)
        # evaluation on test set
        if get_testset_results:
            loss_test = test_mihm(
                model,
                test_dataset_torch_sample,
                survey_weights=survey_weights,
                regularize=(elasticnet or lasso),
                regularization=regularization,
            )
        if return_trajectory:
            traj_epoch = {}
            traj_epoch["train_loss"] = epoch_loss
            if get_testset_results:
                traj_epoch["test_loss"] = loss_test
            if get_l2_lengths:
                traj_epoch["l2"] = l2_lengths

        # save model
        if save_model:
            if epoch % 10 == 0:
                if file_id is not None:
                    save_mihm(model, data_id="{}_{}".format(file_id, epoch))
                else:
                    save_mihm(model, data_id="{}".format(epoch))

        printout = "Epoch {}/{} done!; Training Loss: {};".format(
            epoch + 1, epochs, epoch_loss
        )
        if get_testset_results:
            printout += " Testing Loss: {};".format(loss_test)
        # evaluate on significance
        if evaluate:
            if epoch % eval_epoch == 0:
                if epoch % 30 == 0:
                    quietly = False
                else:
                    quietly = True
                if model.include_bias_focal_predictor:
                    thresholded_value = (
                        model.interactor_bias.cpu().detach().numpy().item(0)
                        * all_heat_dataset.mean_std_dict[all_heat_dataset.interactor][1]
                        + all_heat_dataset.mean_std_dict[all_heat_dataset.interactor][0]
                    )
                else:
                    thresholded_value = 0.0
                regression_summary = eval_mihm(
                    model,
                    train_mihm_dataset,
                    df_orig,
                    regress_cmd,
                    use_stata=use_stata,
                    file_id=file_id,
                    threshold=model.include_bias_focal_predictor,
                    thresholded_value=thresholded_value,
                    interaction_direction=interaction_direction,
                )
                if get_testset_results:
                    assert test_mihm_dataset is not None
                    regression_summary_test = eval_mihm(
                        model,
                        test_mihm_dataset,
                        df_orig,
                        regress_cmd,
                        use_stata=use_stata,
                        file_id=file_id,
                        quietly=quietly,
                        threshold=model.include_bias_focal_predictor,
                        thresholded_value=thresholded_value,
                        interaction_direction=interaction_direction,
                    )
                if return_trajectory:
                    traj_epoch["regression_summary"] = regression_summary
                    printout += " Regression Summary: {};".format(regression_summary)
                    if get_testset_results:
                        traj_epoch["regression_summary_test"] = regression_summary_test
                        printout += " Testset Regression Summary: {};".format(
                            regression_summary_test
                        )
                    # traj_epoch["index_predictions"] = index_predictions
            if early_stop:
                if (
                    regression_summary["interaction term p value"]
                    < early_stop_criterion
                    and regression_summary_test["interaction term p value"]
                    < early_stop_criterion
                    and epoch > stop_after
                ):
                    print(
                        "reached early stopping criterion!!!!!, epoch: {}".format(epoch)
                    )
                    print(printout)
                    break
        trajectory_data.append(traj_epoch)
        print(printout)

    # final evaluation
    if evaluate:
        if model.include_bias_focal_predictor:
            thresholded_value = (
                model.interactor_bias.cpu().detach().numpy().item(0)
                * all_heat_dataset.mean_std_dict[all_heat_dataset.interactor][1]
                + all_heat_dataset.mean_std_dict[all_heat_dataset.interactor][0]
            )
        final_summary = eval_mihm(
            model,
            all_heat_dataset,
            df_orig,
            regress_cmd,
            use_stata=use_stata,
            file_id=file_id + 1,
            quietly=False,
            threshold=model.include_bias_focal_predictor,
            thresholded_value=thresholded_value,
            interaction_direction=interaction_direction,
        )
        print(final_summary)

    if return_trajectory:
        return model, trajectory_data
    else:
        return model


def eval_mihm(
    model: MIHM,
    test_mihm_dataset: MIHMDataset,
    df_orig,
    regress_cmd: str,
    use_stata: bool = True,
    file_id: Union[str, None] = None,
    quietly: bool = True,
    threshold: bool = True,
    thresholded_value: float = 0.0,
    interaction_direction: str = "positive",
):
    test_interaction_predictors = test_mihm_dataset.to_tensor(device=TRAIN_DEVICE)[
        "interaction_predictors"
    ]
    index_predictions = compute_index_prediction(model, test_interaction_predictors)
    try:
        if use_stata:
            interaction_pval, (rsq, adjusted_rsq, rmse), (vif_heat, vif_inter) = (
                evaluate_significance_stata(
                    df_orig.iloc[test_mihm_dataset.df.index].copy(),
                    index_predictions,
                    regress_cmd,
                    data_id=file_id,
                    save_intermediate=True,
                    quietly=quietly,
                    threshold=threshold,
                    thresholded_value=thresholded_value,
                    interaction_direction=interaction_direction,
                )
            )
        else:
            interaction_pval = evaluate_significance(
                df_orig.iloc[test_mihm_dataset.df.index].copy(),
                regress_cmd,
                index_predictions,
            )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        interaction_pval = 0.1
    if use_stata:
        return {
            "interaction term p value": interaction_pval,
            "r squared": rsq,
            "adjusted r squared": adjusted_rsq,
            "Root MSE": rmse,
            "vif heat": vif_heat,
            "vif vul index": vif_inter,
        }
    else:
        return {"interaction term p value": interaction_pval}


def test_mihm(
    model: MIHM,
    test_dataset_torch_sample: MIHMDataset,
    survey_weights: bool = False,
    regularize: bool = False,
    regularization=None,
):
    model.eval()
    if survey_weights:
        mseLoss = nn.MSELoss(reduction="none")
    else:
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
        if regularize:
            reg_loss = sum(
                regularization(p) for p in model.index_prediction_model.parameters()
            )
        if survey_weights:
            assert loss_test.shape[0] == test_dataset_torch_sample["weights"].shape[0]
            loss_test = (loss_test * test_dataset_torch_sample["weights"]).mean()

    return loss_test.item()


def save_mihm(
    model: MIHM,
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Union[str, None] = None,
):
    if data_id is not None:
        model_name = os.path.join(save_dir, f"mihm_model_{data_id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"mihm_model_{num_files}.pt")
    torch.save(model.state_dict(), model_name)
