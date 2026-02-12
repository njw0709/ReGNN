import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import add_constant
from regnn.probe import (
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
)
from .utils import init_stata
from regnn.probe import DataSource


# ---------------------------------------------------------------------------
# Stata-based functions (original)
# ---------------------------------------------------------------------------


def OLS_stata(
    df: pd.DataFrame,
    regress_cmd: str,
    data_source: DataSource = DataSource.TEST,
    quietly: bool = False,
    df_already_moved: bool = False,
) -> OLSModeratedResultsProbe:
    """Original Stata-based significance evaluation"""
    stata = init_stata()

    # move df to stata
    if not df_already_moved:
        stata.pdataframe_to_data(df, force=True)
    stata.run(regress_cmd, quietly=quietly)
    regression_results = stata.get_return()
    eresults = stata.get_ereturn()
    rsq = eresults["e(r2)"]
    adjusted_rsq = eresults["e(r2_a)"]
    rmse = eresults["e(rmse)"]
    interaction_pval = regression_results["r(table)"][3, 1]
    results = OLSModeratedResultsProbe(
        data_source=data_source,
        rsquared=rsq,
        adjusted_rsquared=adjusted_rsq,
        rmse=rmse,
        interaction_pval=interaction_pval,
    )

    return results


def VIF_stata(
    df: pd.DataFrame,
    data_source: DataSource = DataSource.TEST,
    quietly: bool = False,
    df_already_moved: bool = False,
) -> VarianceInflationFactorProbe:
    stata = init_stata()

    # move df to stata
    if not df_already_moved:
        stata.pdataframe_to_data(df, force=True)
    # vif
    stata.run("vif", quietly=quietly)
    vif_results = stata.get_return()
    vif_main = vif_results["r(vif_1)"]
    vif_inter = vif_results["r(vif_2)"]

    return VarianceInflationFactorProbe(
        data_source=data_source,
        vif_main=vif_main,
        vif_interaction=vif_inter,
    )


# ---------------------------------------------------------------------------
# statsmodels helper utilities
# ---------------------------------------------------------------------------


def _resolve_col_name(col: str, rename_dict: dict) -> str:
    """Resolve a column name to its original DataFrame column name.

    ``rename_dict`` maps ``{original_name: renamed_name}``.  If *col*
    matches a *renamed* value the corresponding *original* key is returned.
    Otherwise *col* is returned unchanged.
    """
    for orig, renamed in rename_dict.items():
        if col == renamed:
            return orig
    return col


def _is_categorical_col(col: str, data_readin_config) -> bool:
    """Return True if *col* should be dummy-encoded.

    Checks against ``binary_cols``, ``categorical_cols``, and
    ``ordinal_cols`` in *data_readin_config*.
    """
    return (
        (data_readin_config.binary_cols and col in data_readin_config.binary_cols)
        or (
            data_readin_config.categorical_cols
            and col in data_readin_config.categorical_cols
        )
        or (data_readin_config.ordinal_cols and col in data_readin_config.ordinal_cols)
    )


# ---------------------------------------------------------------------------
# Stata-style summary table builder
# ---------------------------------------------------------------------------


def _build_stata_style_summary(
    model_fit,
    y: pd.Series,
    rsq: float,
    adj_rsq: float,
    rmse: float,
    weights: Optional[np.ndarray],
) -> str:
    """Build a Stata-style regression summary table from a fitted model.

    Always succeeds (does not rely on ``model_fit.summary()`` / patsy).
    The layout mirrors Stata's ``regress`` output as closely as possible.
    """
    from scipy import stats as sp_stats

    n_obs = int(model_fit.nobs)
    df_model = int(model_fit.df_model)
    df_resid = int(model_fit.df_resid)
    model_type = "WLS" if weights is not None else "OLS"

    # --- F-statistic (may not be available on all fits) ---
    try:
        f_stat = float(model_fit.fvalue)
        f_pval = float(model_fit.f_pvalue)
    except Exception:
        f_stat = float("nan")
        f_pval = float("nan")

    # --- ANOVA-like header ---
    ss_model = float(model_fit.ess)
    ss_resid = float(model_fit.ssr)
    ss_total = ss_model + ss_resid
    ms_model = ss_model / df_model if df_model > 0 else float("nan")
    ms_resid = ss_resid / df_resid if df_resid > 0 else float("nan")

    W = 78  # total line width

    header_lines = [
        f"statsmodels {model_type} Regression Results",
        "=" * W,
        f"{'Source':>13s} |{'SS':>14s} {'df':>8s} {'MS':>14s}   {'Number of obs':>16s} = {n_obs:>8d}",
        f"{'-' * 13}-+{'-' * 38}   {'F({}, {})'.format(df_model, df_resid):>16s} = {f_stat:>8.2f}",
        f"{'Model':>13s} | {ss_model:>13.6g} {df_model:>8d} {ms_model:>14.6g}   {'Prob > F':>16s} = {f_pval:>8.4f}",
        f"{'Residual':>13s} | {ss_resid:>13.6g} {df_resid:>8d} {ms_resid:>14.6g}   {'R-squared':>16s} = {rsq:>8.4f}",
        f"{'-' * 13}-+{'-' * 38}   {'Adj R-squared':>16s} = {adj_rsq:>8.4f}",
        f"{'Total':>13s} | {ss_total:>13.6g} {n_obs - 1:>8d} {ss_total / (n_obs - 1) if n_obs > 1 else 0:>14.6g}   {'Root MSE':>16s} = {rmse:>8.5g}",
    ]

    # --- Coefficient table ---
    sep = "-" * W
    col_header = (
        f"{'':>13s} | {'Coefficient':>12s} {'Std. err.':>12s} "
        f"{'t':>9s} {'P>|t|':>8s}   {'[95% conf. interval]':>22s}"
    )
    col_sep = f"{'-' * 13}-+{'-' * (W - 14)}"

    coef_lines = []
    alpha = 0.05
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df_resid) if df_resid > 0 else float("nan")

    for name, coef, se, pv in zip(
        model_fit.params.index,
        model_fit.params,
        model_fit.bse,
        model_fit.pvalues,
    ):
        t_val = coef / se if se != 0 else float("nan")
        ci_lo = coef - t_crit * se
        ci_hi = coef + t_crit * se
        # Truncate long names to fit the column
        display_name = name if len(name) <= 13 else name[:12] + "~"
        coef_lines.append(
            f"{display_name:>13s} | {coef:>12.6g} {se:>12.6g} "
            f"{t_val:>9.2f} {pv:>8.3f}   {ci_lo:>10.6g}  {ci_hi:>10.6g}"
        )

    # --- Assemble ---
    parts = (
        header_lines
        + [sep, col_header, col_sep]
        + coef_lines
        + [sep]
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Design-matrix construction (config-based)
# ---------------------------------------------------------------------------


def build_regression_design_matrix(
    df: pd.DataFrame,
    regression_config,
    data_readin_config,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Optional[np.ndarray]]:
    """Build a design matrix for OLS / WLS from structured configurations.

    Mirrors the variable construction logic of
    :func:`generate_stata_command` but produces a :class:`pandas.DataFrame`
    suitable for *statsmodels* rather than a Stata command string.

    Parameters
    ----------
    df : DataFrame
        DataFrame with **original** column names (e.g. ``dataset.df_orig``),
        with the index-prediction column already appended.
    regression_config : ModeratedRegressionConfig
        Regression specification (focal predictor, outcome, controls, etc.).
    data_readin_config : DataFrameReadInConfig
        Column-type metadata (binary, categorical, continuous, …).

    Returns
    -------
    X : DataFrame
        Design matrix with a ``const`` column prepended.
    y : Series
        Dependent variable.
    interaction_col_names : list[str]
        Names of the interaction columns in *X*.
    weights : ndarray or None
        Survey weights, or ``None`` when not applicable.
    """
    rename_dict = data_readin_config.rename_dict

    def resolve(c: str) -> str:
        return _resolve_col_name(c, rename_dict)

    # --- Outcome ---------------------------------------------------------
    outcome_df = resolve(regression_config.outcome_col)
    y = df[outcome_df].astype(float)

    # --- Build X columns in the same order as Stata command ---------------
    columns: dict = {}
    col_order: List[str] = []

    # (a) Focal predictor
    focal_cfg = regression_config.focal_predictor
    focal_df = resolve(focal_cfg)
    focal_is_cat = _is_categorical_col(focal_cfg, data_readin_config)

    focal_col_names: List[str] = []
    if focal_is_cat:
        dummies = pd.get_dummies(
            df[focal_df], prefix=focal_df, drop_first=True, dtype=float
        )
        focal_col_names = list(dummies.columns)
        for c in focal_col_names:
            columns[c] = dummies[c]
            col_order.append(c)
    else:
        focal_col_names = [focal_df]
        columns[focal_df] = df[focal_df].astype(float)
        col_order.append(focal_df)

    # (b) Interaction terms: focal × index
    index_col = regression_config.index_column_name
    index_cols = index_col if isinstance(index_col, list) else [index_col]

    interaction_col_names: List[str] = []
    for idx_col in index_cols:
        for f_col in focal_col_names:
            inter_name = f"{f_col}x{idx_col}"
            columns[inter_name] = columns[f_col] * df[idx_col].astype(float)
            col_order.append(inter_name)
            interaction_col_names.append(inter_name)

    # (c) Control variables (controlled_cols + moderators when flagged)
    linear_terms: List[str] = list(regression_config.controlled_cols)
    if regression_config.control_moderators:
        mods = regression_config.moderators
        if isinstance(mods, list) and len(mods) > 0 and isinstance(mods[0], list):
            for mod_list in mods:
                linear_terms.extend(mod_list)
        else:
            linear_terms.extend(mods)

    for col in linear_terms:
        col_df = resolve(col)
        if _is_categorical_col(col, data_readin_config):
            dummies = pd.get_dummies(
                df[col_df], prefix=col_df, drop_first=True, dtype=float
            )
            for c in dummies.columns:
                columns[c] = dummies[c]
                col_order.append(c)
        else:
            columns[col_df] = df[col_df].astype(float)
            col_order.append(col_df)

    # Assemble DataFrame and prepend constant
    X = pd.DataFrame(columns)[col_order]
    X = add_constant(X)

    # --- Survey weights --------------------------------------------------
    weights = None
    if data_readin_config.survey_weight_col is not None:
        weight_df = resolve(data_readin_config.survey_weight_col)
        weights = df[weight_df].values.astype(float)

    return X, y, interaction_col_names, weights


# ---------------------------------------------------------------------------
# Config-based statsmodels OLS / WLS
# ---------------------------------------------------------------------------


def OLS_statsmodel_from_config(
    df: pd.DataFrame,
    regression_config,
    data_readin_config,
    data_source: Union[str, DataSource] = DataSource.TEST,
) -> OLSModeratedResultsProbe:
    """Run OLS (or WLS) regression via *statsmodels* using structured configs.

    Properly handles categorical dummy-encoding, focal × index interaction
    terms, and survey probability weights (via WLS with HC1 robust SEs).

    Parameters
    ----------
    df : DataFrame
        DataFrame with original column names, including the index column.
    regression_config : ModeratedRegressionConfig
    data_readin_config : DataFrameReadInConfig
    data_source : str or DataSource
        Label for the data source (e.g. ``"test"``).

    Returns
    -------
    OLSModeratedResultsProbe
        Comprehensive regression results.
    """
    X, y, interaction_col_names, weights = build_regression_design_matrix(
        df, regression_config, data_readin_config
    )

    # Fit model — WLS with robust (sandwich) SEs when survey weights present
    if weights is not None:
        model_fit = WLS(y, X, weights=weights).fit(cov_type="HC1")
    else:
        model_fit = OLS(y, X).fit()

    # Extract scalar statistics
    rsq = float(model_fit.rsquared)
    adj_rsq = float(model_fit.rsquared_adj)
    rmse = float(np.sqrt(model_fit.mse_resid))
    n_obs = int(model_fit.nobs)

    # Interaction term results (first interaction column)
    interaction_pval = float(model_fit.pvalues[interaction_col_names[0]])

    # Build summary string — model_fit.summary() can fail when the manually-
    # constructed column names are not valid patsy term names (patsy's
    # DesignInfo requires names that satisfy its internal invariant checks).
    # Fall back to a Stata-style summary in that case.
    raw_summary = _build_stata_style_summary(model_fit, y, rsq, adj_rsq, rmse, weights)

    return OLSModeratedResultsProbe(
        data_source=data_source,
        status="success",
        message=(
            f"statsmodels {'WLS (HC1)' if weights is not None else 'OLS'} "
            "regression completed successfully."
        ),
        rsquared=rsq,
        adjusted_rsquared=adj_rsq,
        rmse=rmse,
        interaction_pval=interaction_pval,
        coefficients=model_fit.params.tolist(),
        standard_errors=model_fit.bse.tolist(),
        p_values=model_fit.pvalues.tolist(),
        n_observations=n_obs,
        raw_summary=raw_summary,
    )


# ---------------------------------------------------------------------------
# Config-based statsmodels VIF
# ---------------------------------------------------------------------------


def VIF_statsmodel_from_config(
    df: pd.DataFrame,
    regression_config,
    data_readin_config,
    data_source: Union[str, DataSource] = DataSource.TEST,
) -> VarianceInflationFactorProbe:
    """Calculate VIF via *statsmodels* using the design matrix from configs.

    Parameters
    ----------
    df : DataFrame
        DataFrame with original column names, including the index column.
    regression_config : ModeratedRegressionConfig
    data_readin_config : DataFrameReadInConfig
    data_source : str or DataSource

    Returns
    -------
    VarianceInflationFactorProbe
    """
    X, _, interaction_col_names, _ = build_regression_design_matrix(
        df, regression_config, data_readin_config
    )

    col_names = list(X.columns)

    # Focal predictor is the first non-constant column (index 1 after 'const')
    focal_idx = 1
    interaction_idx = col_names.index(interaction_col_names[0])

    vif_main = float(variance_inflation_factor(X.values, focal_idx))
    vif_inter = float(variance_inflation_factor(X.values, interaction_idx))

    return VarianceInflationFactorProbe(
        data_source=data_source,
        vif_main=vif_main,
        vif_interaction=vif_inter,
    )


# ---------------------------------------------------------------------------
# Legacy command-string based statsmodels functions (kept for backward compat)
# ---------------------------------------------------------------------------


def OLS_statsmodel(
    df: pd.DataFrame,
    regress_cmd: str,
    data_source: DataSource = DataSource.TEST,
) -> OLSModeratedResultsProbe:
    """Legacy statsmodels OLS that parses a Stata ``regress`` command string.

    .. note::
        This function performs *simplistic* parsing of the Stata command and
        does **not** handle categorical (``i.``) prefixes, interaction terms
        (``#``), or survey weights.  Prefer
        :func:`OLS_statsmodel_from_config` for full-featured regression.
    """
    # Parse Stata regression command to get variables
    # Example command: "regress y x1 c.x1#c.res_index x2 x3"
    cmd_parts = regress_cmd.split()
    dependent_var = cmd_parts[1]

    # Strip Stata prefixes (c., i.) and exclude interaction terms / weights
    independent_vars = []
    for part in cmd_parts[2:]:
        if "#" in part or part.startswith("["):
            continue
        # Use prefix removal (not lstrip, which strips individual characters)
        clean = part
        for prefix in ("c.", "i."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        independent_vars.append(clean)

    X = df[independent_vars].astype(float)
    X = add_constant(X)
    y = df[dependent_var].astype(float)

    model = OLS(y, X).fit()

    rsq = float(model.rsquared)
    adjusted_rsq = float(model.rsquared_adj)
    rmse = float(np.sqrt(model.mse_resid))

    # Interaction p-value — best-effort positional guess (2nd coefficient)
    interaction_pval = float(model.pvalues.iloc[2]) if len(model.pvalues) > 2 else float("nan")

    return OLSModeratedResultsProbe(
        data_source=data_source,
        rsquared=rsq,
        adjusted_rsquared=adjusted_rsq,
        rmse=rmse,
        interaction_pval=interaction_pval,
        coefficients=model.params.tolist(),
        standard_errors=model.bse.tolist(),
        p_values=model.pvalues.tolist(),
        n_observations=int(model.nobs),
        raw_summary=str(model.summary()),
    )


def VIF_statsmodel(
    df: pd.DataFrame,
    data_source: DataSource = DataSource.TEST,
) -> VarianceInflationFactorProbe:
    """Legacy VIF calculation — accepts a raw DataFrame.

    Prefer :func:`VIF_statsmodel_from_config` for config-based usage.
    """
    X = add_constant(df)

    vif_values = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # index 0 = constant, index 1 = main variable, index 2 = interaction
    vif_main = vif_values[1]
    vif_inter = vif_values[2]

    return VarianceInflationFactorProbe(
        data_source=data_source,
        vif_main=vif_main,
        vif_interaction=vif_inter,
    )
