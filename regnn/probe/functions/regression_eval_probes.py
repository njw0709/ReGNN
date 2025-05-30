import pandas as pd
from typing import Dict, Any, Callable, Optional, TypeVar

from ..registry import register_probe
from ..dataclass.probe_config import RegressionEvalProbeScheduleConfig
from ..dataclass.regression import OLSModeratedResultsProbe

from regnn.train import TrainingHyperParams

ModeratedRegressionConfig = TypeVar("ModeratedRegressionConfig")
MacroConfig = TypeVar("MacroConfig")
from regnn.eval import init_stata  # For initializing Stata connection

# Corrected import path for local_compute_index_prediction:
from .intermediate_index_probes import compute_index_prediction
from ..dataclass.probe_config import (
    FocalPredictorPreProcessOptions,
)
from regnn.data import ReGNNDataset, DataFrameReadInConfig  # Import ReGNNDataset
from regnn.model import ReGNN


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

    # Add linear terms
    linear_terms = regression_config.controlled_cols
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


@register_probe("regression_eval")
def regression_eval_probe(
    model: ReGNN,
    schedule_config: RegressionEvalProbeScheduleConfig,
    data_source_name: str,  # e.g., "train", "test", "validation"
    dataset: ReGNNDataset,  # Changed type hint to ReGNNDataset
    training_hp: TrainingHyperParams,  # For device primarily
    shared_resource_accessor: Callable[[str], Any],
    epoch: int,  # For context, might be used in filenames or reporting
    **kwargs,
) -> Optional[OLSModeratedResultsProbe]:

    current_status = "success"  # Default status
    status_message = None

    print("regression eval...")
    # --- 1. Retrieve MacroConfig & Basic Setup ---
    macro_config: Optional[MacroConfig] = shared_resource_accessor("macro_config")
    if not macro_config:
        # OLSModeratedResultsProbe requires interaction_pval, so provide a default NaN
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

    # --- 3. Prepare DataFrame for Stata ---
    try:
        df_eval = (
            dataset.df_orig.copy()
        )  # Use a copy to avoid modifying the original dataset's df

        # Ensure idx_pred_np aligns with df_eval's index if it's a 1D array
        # Flatten predictions to 1D array
        if idx_pred_np.ndim == 2 and idx_pred_np.shape[1] == 1:
            idx_pred_np_flat = idx_pred_np.ravel()
        elif idx_pred_np.ndim == 1:
            idx_pred_np_flat = idx_pred_np
        else:  # If idx_pred_np has multiple columns, this probe isn't designed for it.
            raise ValueError(
                f"Index predictions have unexpected shape: {idx_pred_np.shape}. Expected 1D array or 2D array with 1 column."
            )

        if len(idx_pred_np_flat) == len(df_eval):
            df_eval[schedule_config.index_column_name] = idx_pred_np_flat
        else:
            # This could happen if dataset.df was filtered AFTER .to_tensor() was prepared based on an older state.
            # Or if dataset.to_tensor() provides moderators for a subset not matching .df
            raise ValueError(
                f"Length of index predictions ({len(idx_pred_np_flat)}) "
                f"does not match DataFrame length ({len(df_eval)}). "
                "Ensure dataset.df and moderators from dataset.to_tensor() are aligned."
            )

    except Exception as e:
        import traceback

        message = f"Failed to prepare DataFrame for Stata in regression_eval_probe: {e}\n{traceback.format_exc()}"
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

        message = f"Failed to create probe-specific ModeratedRegressionConfig: {e}\\n{traceback.format_exc()}"
        print(message)
        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=message,
            interaction_pval=float("nan"),
        )

    # --- 5. Focal Predictor Processing ---
    if (
        schedule_config.focal_predictor_process_options
        and schedule_config.focal_predictor_process_options.threshold
    ):

        fp_opts: FocalPredictorPreProcessOptions = (
            schedule_config.focal_predictor_process_options
        )
        focal_col_name = current_mod_reg_config.focal_predictor

        # Check model and dataset compatibility for this processing step
        model_has_bias_attrs = hasattr(
            model, "include_bias_focal_predictor"
        ) and hasattr(model, "xf_bias")
        dataset_has_req_attrs = (
            hasattr(dataset, "config")
            and hasattr(dataset.config, "focal_predictor")
            and hasattr(dataset, "mean_std_dict")
        )

        if not model_has_bias_attrs:
            print(
                f"Warning (regression_eval_probe): Model missing 'include_bias_focal_predictor' or 'xf_bias' attributes. Skipping focal predictor thresholding value calculation based on model bias."
            )
        if not dataset_has_req_attrs:
            print(
                f"Warning (regression_eval_probe): Dataset missing 'config.focal_predictor' or 'mean_std_dict' attributes. Skipping focal predictor thresholding value calculation based on dataset stats."
            )

        # Calculate thresholded_value for the preprocessor, if model and dataset support it
        # This logic mirrors regnn/macroutils/evaluator.py
        # The fp_opts.thresholded_value will be set and used by fp_opts.create_preprocessor()
        if model_has_bias_attrs and model.include_bias_focal_predictor:
            if dataset_has_req_attrs and focal_col_name in dataset.mean_std_dict:
                mean_val, std_val = dataset.mean_std_dict[focal_col_name]
                # Ensure xf_bias is a scalar tensor before calling .item()
                model_xf_bias_val = model.xf_bias.cpu().detach().numpy()
                if model_xf_bias_val.size == 1:
                    fp_opts.thresholded_value = (
                        model_xf_bias_val.item() * std_val + mean_val
                    )
                else:
                    print(
                        f"Warning (regression_eval_probe): model.xf_bias is not a scalar ({model_xf_bias_val.shape}). Using raw bias value if possible, else 0."
                    )
                    fp_opts.thresholded_value = (
                        model_xf_bias_val.item() if model_xf_bias_val.size == 1 else 0.0
                    )
            else:
                # Fallback if focal predictor stats are not found in dataset or dataset incompatible
                model_xf_bias_val = model.xf_bias.cpu().detach().numpy()
                fp_opts.thresholded_value = (
                    model_xf_bias_val.item() if model_xf_bias_val.size == 1 else 0.0
                )
                if (
                    not (
                        dataset_has_req_attrs
                        and focal_col_name in dataset.mean_std_dict
                    )
                    and dataset_has_req_attrs
                ):
                    print(
                        f"Warning (regression_eval_probe): Stats for focal predictor '{focal_col_name}' not found in dataset.mean_std_dict. Using model's raw xf_bias for thresholding if available."
                    )
        elif (
            model_has_bias_attrs
        ):  # model does not include_bias_focal_predictor, or condition was false
            # If not using model bias, threshold_value might be set directly in config or default to 0 in FocalPredictorPreProcessOptions
            pass  # fp_opts.thresholded_value would use its default or pre-configured value

        # Apply the preprocessing to the focal predictor column in df_eval
        try:
            preprocessor = fp_opts.create_preprocessor()
            if focal_col_name not in df_eval.columns:
                raise ValueError(
                    f"Focal predictor column '{focal_col_name}' not found in the evaluation DataFrame."
                )

            focal_array = df_eval[
                focal_col_name
            ].values.copy()  # Operate on a copy for safety
            thresholded_focal_array = preprocessor(focal_array)
            df_eval[focal_col_name] = thresholded_focal_array
            # print(f"INFO: regression_eval_probe - Applied focal predictor processing to column '{focal_col_name}'.")
        except Exception as e:
            import traceback

            # If focal processing fails, proceed with original data but log a warning/error
            status_message = f"Error during focal predictor processing: {e}\\n{traceback.format_exc()}. Proceeding with original focal predictor data."
            print(f"ERROR (regression_eval_probe): {status_message}")
            # Potentially change status to partial success or keep as success with message
            # For now, we let it proceed and the error is logged.

    # --- 6. Generate or Get Stata Command ---
    if schedule_config.regress_cmd:
        regress_cmd = schedule_config.regress_cmd
    else:
        try:
            regress_cmd = generate_stata_command(
                df_readin_config, current_mod_reg_config
            )
        except Exception as e:
            import traceback

            return OLSModeratedResultsProbe(
                data_source=data_source_name,
                status="failure",
                message=f"Failed to generate Stata command for regression_eval_probe: {e}\\n{traceback.format_exc()}",
                interaction_pval=float("nan"),
            )

    # --- 7. Execute Stata Regression & Adapt Results ---
    try:
        stata = init_stata()
        stata_quietly_flag = schedule_config.probe_params.get("stata_quietly", True)

        stata.pdataframe_to_data(df_eval, force=True)
        stata.run(regress_cmd, quietly=stata_quietly_flag)

        stata_return = stata.get_return()
        stata_ereturns = stata.get_ereturn()

        if not stata_return or "r(table)" not in stata_return:
            raise ValueError("Stata did not return 'r(table)' with regression results.")
        if not stata_ereturns:
            raise ValueError(
                "Stata did not return e-class results (e.g., e(r2), e(N))."
            )

        # pystata r(table) is typically a matrix where rows are variables and columns are stats (b, se, t, pvalue, ll, ul)
        # Row names (variable names Stata used) are in stata_return["_rowname"]
        # Column names for r(table) are typically stata_return["_colname"] = ["b", "se", "t", "pvalue", "ll", "ul"]

        r_table_matrix = stata_return[
            "r(table)"
        ]  # This should be (num_variables x num_stats)
        r_table_variable_names = stata_return[
            "_rowname"
        ]  # List of variable names as rows in r_table_matrix
        # stata_col_names = stata_return["_colname"] # Should be ["b", "se", "t", "pvalue", "ll", "ul"]

        # Verify expected column indices for b, se, pvalue from _colname or assume standard Stata order
        # Standard order: b (0), se (1), t (2), pvalue (3), ll (4), ul (5)
        COEF_IDX, SE_IDX, PVAL_IDX = 0, 1, 3

        coefficients: Dict[str, float] = {}
        standard_errors: Dict[str, float] = {}
        p_values_dict: Dict[str, float] = {}

        if r_table_matrix.shape[0] != len(r_table_variable_names):
            raise ValueError(
                f"Mismatch between r(table) matrix rows ({r_table_matrix.shape[0]}) and _rowname count ({len(r_table_variable_names)})."
            )
        if (
            r_table_matrix.shape[1] < PVAL_IDX + 1
        ):  # Ensure PVAL_IDX is a valid column index
            raise ValueError(
                f"r(table) matrix does not have enough columns for p-values. Shape: {r_table_matrix.shape}"
            )

        for i, var_name in enumerate(r_table_variable_names):
            coefficients[var_name] = r_table_matrix[
                i, COEF_IDX
            ]  # Stat for variable i, column COEF_IDX
            standard_errors[var_name] = r_table_matrix[
                i, SE_IDX
            ]  # Stat for variable i, column SE_IDX
            p_values_dict[var_name] = r_table_matrix[
                i, PVAL_IDX
            ]  # Stat for variable i, column PVAL_IDX

        r_squared = stata_ereturns.get("e(r2)")
        adj_r_squared = stata_ereturns.get("e(r2_a)")
        rmse = stata_ereturns.get("e(rmse)")
        n_observations = int(stata_ereturns.get("e(N)", 0))

        focal_var_stata = current_mod_reg_config.focal_predictor
        index_var_stata = current_mod_reg_config.index_column_name

        interaction_term_key = None
        interaction_pval = float("nan")

        possible_interaction_keys = [
            key
            for key in r_table_variable_names
            if focal_var_stata in key and index_var_stata in key and "#" in key
        ]
        if possible_interaction_keys:
            interaction_term_key = possible_interaction_keys[0]
            if interaction_term_key in p_values_dict:
                interaction_pval = p_values_dict[interaction_term_key]
            else:
                # This case should ideally not happen if possible_interaction_keys found one from r_table_variable_names
                print(
                    f"Warning (regression_eval_probe): Matched interaction key '{interaction_term_key}' not found in parsed p_values_dict."
                )
        else:
            # Fallback: try a common Stata pattern if direct search fails
            # This is less robust as it doesn't account for Stata's potential i/c/o prefixes fully
            constructed_interaction_pattern = f"c.{focal_var_stata}#c.{index_var_stata}"
            if constructed_interaction_pattern in p_values_dict:
                interaction_pval = p_values_dict[constructed_interaction_pattern]
                interaction_term_key = (
                    constructed_interaction_pattern  # Found it this way
                )
            else:
                # Another common pattern without c. prefix if they are continuous by default
                constructed_interaction_pattern_no_c = (
                    f"{focal_var_stata}#{index_var_stata}"
                )
                if constructed_interaction_pattern_no_c in p_values_dict:
                    interaction_pval = p_values_dict[
                        constructed_interaction_pattern_no_c
                    ]
                    interaction_term_key = constructed_interaction_pattern_no_c
                else:
                    print(
                        f"Warning (regression_eval_probe): Could not definitively identify interaction term among {r_table_variable_names} for p-value using smart search or common patterns. Setting to NaN."
                    )

        raw_summary_str = f"Stata Regression Output Summary:\nR-squared: {r_squared if r_squared is not None else 'N/A'}\nAdj. R-squared: {adj_r_squared if adj_r_squared is not None else 'N/A'}\nRMSE: {rmse if rmse is not None else 'N/A'}\nN: {n_observations}\nInteraction Term ('{interaction_term_key if interaction_term_key else 'N/A'}') P-value: {interaction_pval:.4f}\n\nCoefficients:\n"
        for var, coef in coefficients.items():
            raw_summary_str += f"  {var}: {coef:.4f} (SE: {standard_errors.get(var, float('nan')):.4f}, P>|t|: {p_values_dict.get(var, float('nan')):.4f})\n"

        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="success",
            message=status_message or "Successfully executed Stata regression.",
            interaction_pval=interaction_pval,
            coefficients=coefficients,
            standard_errors=standard_errors,
            p_values=p_values_dict,
            n_observations=n_observations,
            rsquared=r_squared,
            adjusted_rsquared=adj_r_squared,
            rmse=rmse,
            raw_summary=raw_summary_str,
        )

    except Exception as e:
        import traceback

        return OLSModeratedResultsProbe(
            data_source=data_source_name,
            status="failure",
            message=f"Error during Stata execution or result parsing: {e}\\n{traceback.format_exc()}",
            interaction_pval=float("nan"),
        )

    # Fallback, should not be reached if try/except covers all paths
    return OLSModeratedResultsProbe(
        data_source=data_source_name,
        status="failure",
        message="Reached unexpected end of regression_eval_probe.",
        interaction_pval=float("nan"),
    )
