import torch
import torch.nn as nn
from regnn.model.regnn import ReGNN
from regnn.data.dataset import ReGNNDataset
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import traceback
from .eval import compute_index_prediction, evaluate_significance_stata


def eval_regnn(
    model: ReGNN,
    test_regnn_dataset: ReGNNDataset,
    df_orig: Optional[pd.DataFrame],
    regress_cmd: str,
    use_stata: bool = True,
    file_id: Optional[str] = None,
    quietly: bool = True,
    threshold: bool = True,
    thresholded_value: float = 0.0,
    interaction_direction: str = "positive",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Evaluate ReGNN model performance"""
    test_moderators = test_regnn_dataset.to_tensor(device=device)["moderators"]
    index_predictions = compute_index_prediction(model, test_moderators)

    try:
        if use_stata:
            interaction_pval, (rsq, adjusted_rsq, rmse), (vif_heat, vif_inter) = (
                evaluate_significance_stata(
                    df_orig.iloc[test_regnn_dataset.df.index].copy(),
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
            raise NotImplementedError

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


def test_regnn(
    model: ReGNN,
    test_dataset_torch_sample: Dict[str, torch.Tensor],
    survey_weights: bool = False,
    regularize: bool = False,
    regularization: Optional[nn.Module] = None,
) -> float:
    """Test ReGNN model and return loss"""
    model.eval()
    if survey_weights:
        mseLoss = nn.MSELoss(reduction="none")
    else:
        mseLoss = nn.MSELoss()
    model.return_logvar = False

    with torch.no_grad():
        predicted_epi = model(
            test_dataset_torch_sample["moderators"],
            test_dataset_torch_sample["focal_predictor"],
            test_dataset_torch_sample["controlled_predictors"],
        )
        loss_test = mseLoss(
            predicted_epi, torch.unsqueeze(test_dataset_torch_sample["outcome"], 1)
        )
        if regularize and regularization is not None:
            reg_loss = sum(
                regularization(p) for p in model.index_prediction_model.parameters()
            )
        if survey_weights:
            assert loss_test.shape[0] == test_dataset_torch_sample["weights"].shape[0]
            loss_test = (loss_test * test_dataset_torch_sample["weights"]).mean()

    return loss_test.item()
