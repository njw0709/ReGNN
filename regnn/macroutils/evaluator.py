import torch
import torch.nn as nn
import pandas as pd
from regnn.model.regnn import ReGNN
from regnn.data.dataset import ReGNNDataset
from typing import Dict, Optional, Tuple, Callable, Any
from regnn.eval.base import RegressionEvalOptions
from regnn.model.base import ReGNNConfig
from regnn.train.base import TrainingHyperParams
from .utils import compute_index_prediction
from regnn.eval.eval import (
    OLS_stata,
    VIF_stata,
    OLS_statsmodel,
    VIF_statsmodel,
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
    eval_regnn as low_level_eval_regnn,
)


def eval_regnn(
    model: ReGNN,
    eval_regnn_dataset: ReGNNDataset,
    eval_options: RegressionEvalOptions,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_source: str = "test",
) -> Tuple[OLSModeratedResultsProbe, VarianceInflationFactorProbe]:
    """Evaluate ReGNN model performance using probes (original function)

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
    ols_func: Callable[..., OLSModeratedResultsProbe] = (
        OLS_stata if eval_options.evaluation_function == "stata" else OLS_statsmodel
    )
    vif_func: Callable[..., VarianceInflationFactorProbe] = (
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


def get_thresholded_value(model: ReGNN, dataset: ReGNNDataset) -> float:
    """Get thresholded value for evaluation based on focal predictor's mean and std"""
    if model.include_bias_focal_predictor:
        # Ensure dataset.config and mean_std_dict are correctly populated
        if dataset.config.focal_predictor in dataset.mean_std_dict:
            mean_val, std_val = dataset.mean_std_dict[dataset.config.focal_predictor]
            return (
                model.interactor_bias.cpu().detach().numpy().item(0) * std_val
                + mean_val
            )
        else:
            # Fallback or warning if focal predictor stats are not found
            print(
                f"Warning: Stats for focal predictor '{dataset.config.focal_predictor}' not found in dataset.mean_std_dict. Using raw bias."
            )
            return model.interactor_bias.cpu().detach().numpy().item(0)
    return 0.0


def get_regression_summary(
    model: ReGNN,
    dataset: ReGNNDataset,
    df_orig_with_predictions: pd.DataFrame,
    training_config: TrainingHyperParams,
    regnn_model_config: ReGNNConfig,
    eval_options: RegressionEvalOptions,
    thresholded_value: float = 0.0,
    quietly: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model using regression and return a summary dictionary.
    This function is analogous to the eval_regnn that was previously part of the trainer logic.
    It uses the low_level_eval_regnn from regnn.eval.eval.
    """

    summary_dict: Dict[str, Any] = low_level_eval_regnn(
        df=df_orig_with_predictions,
        regress_cmd=training_config.regress_cmd,
        file_id=(
            training_config.file_id if hasattr(training_config, "file_id") else None
        ),
        quietly=quietly,
        threshold=regnn_model_config.include_bias_focal_predictor,
        thresholded_value=thresholded_value,
        interaction_direction=regnn_model_config.interaction_direction,
        outcome_col=dataset.config.outcome,
        focal_predictor_col=dataset.config.focal_predictor,
        index_col=eval_options.index_column_name,
        controlled_vars_cols=dataset.config.controlled_predictors,
        use_stata=eval_options.evaluation_function == "stata",
    )
    return summary_dict
