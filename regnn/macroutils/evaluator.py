import torch
from regnn.model.regnn import ReGNN
from regnn.data.dataset import ReGNNDataset
from typing import Tuple, Callable
from .utils import compute_index_prediction
from regnn.eval import (
    OLS_stata,
    VIF_stata,
    OLS_statsmodel,
    VIF_statsmodel,
    RegressionEvalOptions,
)
from regnn.probe import OLSModeratedResultsProbe, VarianceInflationFactorProbe


def regression_eval_regnn(
    model: ReGNN,
    eval_regnn_dataset: ReGNNDataset,
    eval_options: RegressionEvalOptions,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_source: str = "validate",
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
    # Calculate thresholded value if needed
    if (
        eval_options.focal_predictor_process_options.threshold
        and model.include_bias_focal_predictor
    ):
        focal_predictor = eval_regnn_dataset.config.focal_predictor
        if focal_predictor in eval_regnn_dataset.mean_std_dict:
            mean_val, std_val = eval_regnn_dataset.mean_std_dict[focal_predictor]
            thresholded_value = (
                model.interactor_bias.cpu().detach().numpy().item(0) * std_val
                + mean_val
            )
            # Update the thresholded_value in the options
            eval_options.focal_predictor_process_options.thresholded_value = (
                thresholded_value
            )
        else:
            # Fallback if focal predictor stats are not found
            print(
                f"Warning: Stats for focal predictor '{focal_predictor}' not found in dataset.mean_std_dict. Using raw bias."
            )
            eval_options.focal_predictor_process_options.thresholded_value = (
                model.interactor_bias.cpu().detach().numpy().item(0)
            )

    # Get model predictions
    test_moderators = eval_regnn_dataset.to_tensor(device=device)["moderators"]
    index_predictions = compute_index_prediction(model, test_moderators)

    # Get test data
    test_df = eval_regnn_dataset.df_orig.copy()

    # Get preprocessor function
    preprocessor = eval_options.focal_predictor_process_options.create_preprocessor()

    # Process the focal predictor (index predictions) if thresholding is enabled
    if eval_options.focal_predictor_process_options.threshold:
        focal_predictor = eval_regnn_dataset.config.focal_predictor
        # Convert dataframe column to numpy array, apply preprocessor, then update the dataframe
        focal_array = test_df[focal_predictor].values
        thresholded_focal = preprocessor(focal_array)
        test_df[focal_predictor] = thresholded_focal

    # append index_predictions to test_df
    col_name = eval_options.index_column_name
    test_df[col_name] = index_predictions

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
        df_already_moved=True,
    )

    return ols_results, vif_results
