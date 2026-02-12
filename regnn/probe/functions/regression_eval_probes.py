from typing import Any, Callable, Optional, TypeVar

import pandas as pd

from regnn.train import TrainingHyperParams

from ..registry import register_probe
from ..dataclass.probe_config import RegressionEvalProbeScheduleConfig
from ..dataclass.regression import OLSModeratedResultsProbe, ModeratedRegressionConfig


MacroConfig = TypeVar("MacroConfig")

# Corrected import path for local_compute_index_prediction:
from .intermediate_index_probes import compute_index_prediction

from regnn.data import ReGNNDataset, DataFrameReadInConfig
from regnn.model import ReGNN


# ---------------------------------------------------------------------------
# Stata command generation
# ---------------------------------------------------------------------------


def generate_stata_command(
    data_readin_config: DataFrameReadInConfig,
    regression_config: ModeratedRegressionConfig,
) -> str:
    """
    Generate a Stata regression command based on the configuration.
    Uses the rename_dict from data_readin_config to map internal column names to Stata variable names.

    Returns:
        str: The complete Stata regression command
    """

    # Helper function to get the correct variable name
    def get_var_name(col: str) -> str:
        # If col is a value in rename_dict, return its key
        # If col is a key in rename_dict, return the original name
        # Otherwise return the original name
        rename_dict = data_readin_config.rename_dict
        if col in rename_dict.values():
            # Find the key for this value
            for key, value in rename_dict.items():
                if value == col:
                    return key
        return col

    # Start with basic regression command
    cmd_parts = ["regress"]

    # Add outcome and focal predictor with interaction
    outcome = get_var_name(regression_config.outcome_col)
    focal = get_var_name(regression_config.focal_predictor)

    # Add outcome and focal predictor with interaction
    cmd_parts.append(outcome)

    # Add appropriate prefix to focal predictor based on its type
    if (
        data_readin_config.binary_cols
        and regression_config.focal_predictor in data_readin_config.binary_cols
    ) or (
        data_readin_config.categorical_cols
        and regression_config.focal_predictor in data_readin_config.categorical_cols
    ):
        cmd_parts.append(f"i.{focal}")
    else:
        cmd_parts.append(f"c.{focal}")

    # Add summary index and focal predictor interactions
    cmd_parts.append(f"c.{focal}#c.{regression_config.index_column_name}")

    # Add linear terms (copy to avoid mutating the config's controlled_cols list)
    linear_terms = list(regression_config.controlled_cols)
    if regression_config.control_moderators:
        linear_terms += regression_config.moderators
    for col in linear_terms:
        col_name = get_var_name(col)
        # Check if the column is binary or categorical in the data config
        if data_readin_config.binary_cols and col in data_readin_config.binary_cols:
            cmd_parts.append(f"i.{col_name}")
        elif (
            data_readin_config.categorical_cols
            and col in data_readin_config.categorical_cols
        ):
            cmd_parts.append(f"i.{col_name}")
        elif data_readin_config.ordinal_cols and col in data_readin_config.ordinal_cols:
            cmd_parts.append(f"i.{col_name}")
        else:
            assert col in data_readin_config.continuous_cols
            cmd_parts.append(f"c.{col_name}")

    # Add survey weights if specified
    if data_readin_config.survey_weight_col is not None:
        weight_col = get_var_name(data_readin_config.survey_weight_col)
        cmd_parts.append(f"[pweight={weight_col}]")

    return " ".join(cmd_parts)


# ---------------------------------------------------------------------------
# Backend-specific regression runners
# ---------------------------------------------------------------------------


def _run_stata_regression(
    df_eval: pd.DataFrame,
    regress_cmd: Optional[str],
    data_readin_config: DataFrameReadInConfig,
    regression_config: ModeratedRegressionConfig,
    data_source_name: str,
    status_message: Optional[str] = None,
) -> OLSModeratedResultsProbe:
    """Execute regression via Stata and return probe results."""
    try:
        from regnn.eval import init_stata

        stata = init_stata()

        # Auto-generate command when not provided
        if regress_cmd is None:
            regress_cmd = generate_stata_command(data_readin_config, regression_config)

        stata.pdataframe_to_data(df_eval, force=True)
        stata.run(regress_cmd)

        stata_return = stata.get_return()
        stata_ereturns = stata.get_ereturn()

        if not stata_return or "r(table)" not in stata_return:
            raise ValueError("Stata did not return 'r(table)' with regression results.")
        if not stata_ereturns:
            raise ValueError(
                "Stata did not return e-class results (e.g., e(r2), e(N))."
            )

        r_table_matrix = stata_return["r(table)"]

        # Standard order: b (0), se (1), t (2), pvalue (3), ll (4), ul (5)
        COEF_IDX, SE_IDX, PVAL_IDX = 0, 1, 3

        coefficients = r_table_matrix[COEF_IDX, :].tolist()
        standard_errors = r_table_matrix[SE_IDX, :].tolist()
        p_values = r_table_matrix[PVAL_IDX, :].tolist()

        r_squared = stata_ereturns.get("e(r2)")
        adj_r_squared = stata_ereturns.get("e(r2_a)")
        rmse = stata_ereturns.get("e(rmse)")
        n_observations = int(stata_ereturns.get("e(N)", 0))

        interaction_pval = p_values[1]
        interaction_coef = coefficients[1]

        raw_summary_str = (
            f"Stata Regression Output Summary:\n"
            f"R-squared: {r_squared if r_squared is not None else 'N/A'}\n"
            f"Adj. R-squared: {adj_r_squared if adj_r_squared is not None else 'N/A'}\n"
            f"RMSE: {rmse if rmse is not None else 'N/A'}\n"
            f"Interaction Coef: {interaction_coef:.4f}\n"
            f"P-value: {interaction_pval:.4f}"
        )

        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="success",
            message=status_message or "Successfully executed Stata regression.",
            interaction_pval=interaction_pval,
            coefficients=coefficients,
            standard_errors=standard_errors,
            p_values=p_values,
            n_observations=n_observations,
            rsquared=r_squared,
            adjusted_rsquared=adj_r_squared,
            rmse=rmse,
            raw_summary=raw_summary_str,
        )

    except Exception as e:
        import traceback

        message = f"Error during Stata execution or result parsing: {e}\n{traceback.format_exc()}"
        print(message)

        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )


def _run_statsmodels_regression(
    df_eval: pd.DataFrame,
    regression_config: ModeratedRegressionConfig,
    data_readin_config: DataFrameReadInConfig,
    data_source_name: str,
    status_message: Optional[str] = None,
) -> OLSModeratedResultsProbe:
    """Execute regression via statsmodels and return probe results."""
    try:
        from regnn.eval import OLS_statsmodel_from_config

        result = OLS_statsmodel_from_config(
            df=df_eval,
            regression_config=regression_config,
            data_readin_config=data_readin_config,
            data_source=data_source_name,
        )

        # Print summary table to stdout (mirrors Stata's automatic output)
        if result.raw_summary:
            print(result.raw_summary)

        # Preserve any status message from earlier processing steps
        if status_message:
            result.message = status_message
        return result

    except Exception as e:
        import traceback

        message = f"Error during statsmodels regression: {e}\n{traceback.format_exc()}"
        print(message)

        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


@register_probe("regression_eval")
def regression_eval_probe(
    model: ReGNN,
    schedule_config: RegressionEvalProbeScheduleConfig,
    data_source_name: str,  # e.g., "train", "test", "validation"
    dataset: ReGNNDataset,
    training_hp: TrainingHyperParams,  # For device primarily
    shared_resource_accessor: Callable[[str], Any],
    epoch: int,  # For context, might be used in filenames or reporting
    **kwargs,
) -> Optional[OLSModeratedResultsProbe]:

    status_message = None

    # --- 1. Retrieve MacroConfig & Basic Setup ---
    macro_config: Optional[MacroConfig] = shared_resource_accessor("macro_config")
    if not macro_config:
        message = "MacroConfig not found in shared_resources. This is essential for regression evaluation."
        print(message)
        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )

    # Determine device from training_hp, default to "cpu"
    device_to_use = getattr(training_hp, "device", "cpu")

    # --- 2. Compute Index Predictions ---
    try:
        dataset_tensors = dataset.to_tensor(device=device_to_use)
        interaction_predictors_tensor = dataset_tensors.get("moderators")
        idx_pred_np = compute_index_prediction(model, interaction_predictors_tensor)
    except Exception as e:
        import traceback

        message = f"Failed to compute index predictions for regression_eval_probe: {e}\n{traceback.format_exc()}"
        print(message)
        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )

    # --- 3. Prepare DataFrame for Evaluation ---
    try:
        df_eval = (
            dataset.df_orig.copy()
        )  # Use a copy to avoid modifying the original dataset's df
        # Flatten predictions to 1D array
        if idx_pred_np.ndim == 2 and idx_pred_np.shape[1] == 1:
            idx_pred_np_flat = idx_pred_np.ravel()
        elif idx_pred_np.ndim == 1:
            idx_pred_np_flat = idx_pred_np
        else:
            raise ValueError(
                f"Index predictions have unexpected shape: {idx_pred_np.shape}. "
                "Expected 1D array or 2D array with 1 column."
            )

        if len(idx_pred_np_flat) == len(df_eval):
            df_eval[schedule_config.index_column_name] = idx_pred_np_flat
        else:
            raise ValueError(
                f"Length of index predictions ({len(idx_pred_np_flat)}) "
                f"does not match DataFrame length ({len(df_eval)}). "
                "Ensure dataset.df and moderators from dataset.to_tensor() are aligned."
            )

    except Exception as e:
        import traceback

        message = f"Failed to prepare DataFrame for regression_eval_probe: {e}\n{traceback.format_exc()}"
        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )

    # --- 4. Prepare Probe-Specific ModeratedRegressionConfig ---
    df_readin_config: DataFrameReadInConfig = macro_config.read_config
    mod_reg_config_base: ModeratedRegressionConfig = macro_config.regression

    try:
        current_mod_reg_config_data = mod_reg_config_base.model_dump(exclude_none=True)
        current_mod_reg_config_data["index_column_name"] = (
            schedule_config.index_column_name
        )
        current_mod_reg_config = ModeratedRegressionConfig(
            **current_mod_reg_config_data
        )
    except Exception as e:
        import traceback

        message = f"Failed to create probe-specific ModeratedRegressionConfig: {e}\n{traceback.format_exc()}"
        print(message)
        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )

    # --- 5. Focal Predictor Processing ---
    if (
        schedule_config.focal_predictor_preprocess_options
        and schedule_config.focal_predictor_preprocess_options.threshold
    ):
        focal_col_name = current_mod_reg_config.focal_predictor
        assert (
            model.include_bias_focal_predictor
        ), "Warning (regression_eval_probe, thresholding): Model is set to false on 'include_bias_focal_predictor'"

        try:
            if focal_col_name in dataset.mean_std_dict:
                mean_val, std_val = dataset.mean_std_dict[focal_col_name]
                model_xf_bias_val = model.xf_bias.cpu().detach().numpy()
                if model_xf_bias_val.size == 1:
                    schedule_config.focal_predictor_preprocess_options.thresholded_value = (
                        model_xf_bias_val.item() * std_val + mean_val
                    )
                else:
                    raise AssertionError("Bias value is not size 1")
            preprocessor = (
                schedule_config.focal_predictor_preprocess_options.create_preprocessor()
            )

            focal_array = df_eval[focal_col_name].values.copy()
            thresholded_focal_array = preprocessor(focal_array)
            df_eval[focal_col_name] = thresholded_focal_array
        except Exception as e:
            import traceback

            status_message = (
                f"Error during focal predictor processing: {e}\n"
                f"{traceback.format_exc()}. Proceeding with original focal predictor data."
            )
            print(f"ERROR (regression_eval_probe): {status_message}")

    # --- 6. Execute Regression (Stata or statsmodels) ---
    if schedule_config.evaluation_function == "statsmodels":
        return _run_statsmodels_regression(
            df_eval=df_eval,
            regression_config=current_mod_reg_config,
            data_readin_config=df_readin_config,
            data_source_name=data_source_name,
            status_message=status_message,
        )
    else:  # "stata" (default)
        return _run_stata_regression(
            df_eval=df_eval,
            regress_cmd=schedule_config.regress_cmd,
            data_readin_config=df_readin_config,
            regression_config=current_mod_reg_config,
            data_source_name=data_source_name,
            status_message=status_message,
        )
