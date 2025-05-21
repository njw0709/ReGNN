import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from regnn.probe import (
    OLSModeratedResultsProbe,
    VarianceInflationFactorProbe,
)
from .utils import init_stata


def OLS_stata(
    df: pd.DataFrame,
    regress_cmd: str,
    data_source: str = "test",
    quietly: bool = True,
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


def OLS_statsmodel(
    df: pd.DataFrame,
    regress_cmd: str,
    data_source: str = "test",
) -> OLSModeratedResultsProbe:
    """Python statsmodels version of OLS regression"""
    # Parse Stata regression command to get variables
    # Example command: "regress y x1 c.x1#c.res_index x2 x3"
    cmd_parts = regress_cmd.split()
    dependent_var = cmd_parts[1]
    focal_predictor = cmd_parts[2]

    # Get other control variables (excluding the interaction term notation)
    control_vars = [var for var in cmd_parts[2:] if "#" not in var]

    # Prepare X and y for regression
    X = df[control_vars]
    X = add_constant(X)
    y = df[dependent_var]

    # Fit regression model
    model = OLS(y, X).fit()

    # Get regression statistics
    rsq = model.rsquared
    adjusted_rsq = model.rsquared_adj
    rmse = np.sqrt(model.mse_resid)

    # Get interaction term p-value (it's the 3rd coefficient - after constant, focal predictor, and index)
    interaction_pval = model.pvalues[2]

    return OLSModeratedResultsProbe(
        data_source=data_source,
        rsquared=rsq,
        adjusted_rsquared=adjusted_rsq,
        rmse=rmse,
        interaction_pval=interaction_pval,
    )


def VIF_stata(
    df: pd.DataFrame,
    data_source: str = "test",
    quietly: bool = True,
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


def VIF_statsmodel(
    df: pd.DataFrame,
    data_source: str = "test",
) -> VarianceInflationFactorProbe:
    """Calculate Variance Inflation Factor using statsmodels"""
    # Add constant to the dataframe for VIF calculation
    X = add_constant(df)

    # Calculate VIF for each variable
    vif_values = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Get VIF for main variable (index 1) and interaction term (index 2)
    # Note: index 0 is the constant term
    vif_main = vif_values[1]
    vif_inter = vif_values[2]

    return VarianceInflationFactorProbe(
        data_source=data_source,
        vif_main=vif_main,
        vif_interaction=vif_inter,
    )
