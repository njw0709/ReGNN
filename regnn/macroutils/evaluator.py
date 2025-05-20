import torch
import torch.nn as nn
from regnn.model.regnn import ReGNN
from regnn.data.dataset import ReGNNDataset
from typing import Dict, Optional, Tuple, Callable
from regnn.eval.base import EvaluationOptions
from .utils import compute_index_prediction
from regnn.eval.eval import (
    OLS_stata,
    VIF_stata,
    OLS_statsmodel,
    VIF_statsmodel,
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
)


def eval_regnn(
    model: ReGNN,
    eval_regnn_dataset: ReGNNDataset,
    eval_options: EvaluationOptions,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_source: str = "test",
) -> Tuple[OLSModeratedResultsProbe, VarianceInflationFactorProbe]:
    """Evaluate ReGNN model performance

    Args:
        model: ReGNN model to evaluate
        eval_regnn_dataset: Test dataset for evaluation
        eval_options: Evaluation configuration options
        device: Device to run model on
        data_source: Source of the data for evaluation

    Returns:
        Tuple containing:
        - OLSModeratedResultsProbe: Results from OLS regression
        - VarianceInflationFactorProbe: Results from VIF calculation
    """
    # Get model predictions
    test_moderators = eval_regnn_dataset.to_tensor(device=device)["moderators"]
    index_predictions = compute_index_prediction(model, test_moderators)

    # Get test data
    test_df = eval_regnn_dataset.df_orig.copy()
    # append index_predictions to test_df
    col_name = eval_options.index_column_name
    test_df[col_name] = index_predictions.cpu().numpy()

    # Assign evaluation functions based on evaluation_function
    ols_func: Callable = (
        OLS_stata if eval_options.evaluation_function == "stata" else OLS_statsmodel
    )
    vif_func: Callable = (
        VIF_stata if eval_options.evaluation_function == "stata" else VIF_statsmodel
    )

    # Run OLS regression
    ols_results = ols_func(
        df=test_df,
        regress_cmd=eval_options.regress_cmd,
        data_source=data_source,
    )

    # Calculate VIF
    vif_results = vif_func(
        df=test_df,
        data_source=data_source,
    )

    return ols_results, vif_results


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
