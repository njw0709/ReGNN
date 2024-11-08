from mihm.model.mihm import IndexPredictionModel, MIHM
import matplotlib.pyplot as plt
from typing import Union, Sequence
import torch
import numpy as np
import pandas as pd
import os
import stata_setup
import shap
import matplotlib.pyplot as plt
from .constants import TEMP_DIR
import statsmodels.api as sm
import statsmodels.formula.api as smf


def init_stata():
    stata_setup.config("/usr/local/stata17", "mp")


def init_shap():
    shap.initjs()


def compute_index_prediction(
    model: MIHM, interaction_predictors: torch.Tensor
) -> np.ndarray:

    index_model = model.index_prediction_model
    index_model.to(interaction_predictors.device).eval()
    if index_model.vae:
        index_prediction, log_var = index_model(interaction_predictors)
    else:
        index_prediction = index_model(interaction_predictors)
    index_prediction = index_prediction.detach().cpu().numpy()

    return index_prediction


def evaluate_significance_stata(
    df_orig: pd.DataFrame,
    index_predictions: np.ndarray,
    regress_cmd: str,
    save_dir: str = os.path.join(TEMP_DIR, "data"),
    data_id: Union[str, None] = None,
    save_intermediate: bool = False,
    quietly: bool = True,
    threshold: bool = True,
    thresholded_value: float = 0.0,
    interaction_direction: str = "positive",
):
    init_stata()
    from pystata import stata

    if save_intermediate:
        if data_id is not None:
            save_path = os.path.join(save_dir, f"index_prediction_{data_id}.dta")
        else:
            num_files = len(os.listdir(save_dir))
            save_path = os.path.join(save_dir, f"index_prediction_{num_files}.dta")
    else:
        save_path = os.path.join(save_dir, f"index_prediction_{data_id}.dta")
    # save to file
    if interaction_direction == "positive":
        output_index_name = "res_index"
    elif interaction_direction == "negative":
        output_index_name = "vul_index"
    else:
        raise ValueError("interaction Direction must either be positive or negative!!")

    df_orig[output_index_name] = index_predictions
    if threshold:
        # get focal predictor
        focal_predictor = regress_cmd.split(" ")[2]
        df_orig[focal_predictor] = df_orig[focal_predictor] - thresholded_value
        df_orig.loc[(df_orig[focal_predictor] < 0), focal_predictor] = 0.0
    df_orig.to_stata(save_path, write_index=False)

    # run regression and get significance
    load_cmd = f"use {save_path}, clear"
    stata.run(load_cmd, quietly=True)
    stata.run(regress_cmd, quietly=quietly)
    regression_results = stata.get_return()
    eresults = stata.get_ereturn()
    rsq = eresults["e(r2)"]
    adjusted_rsq = eresults["e(r2_a)"]
    rmse = eresults["e(rmse)"]
    interaction_pval = regression_results["r(table)"][3, 1]

    # vif
    stata.run("vif", quietly=quietly)
    vif_results = stata.get_return()
    vif_heat = vif_results["r(vif_1)"]
    vif_inter = vif_results["r(vif_2)"]

    return interaction_pval, (rsq, adjusted_rsq, rmse), (vif_heat, vif_inter)


def draw_margins_plot(
    data_path: str,
    fig_dir: str = os.path.join(TEMP_DIR, "figures"),
    data_id: Union[str, None] = None,
):
    init_stata()
    from pystata import stata

    # load data
    load_cmd = f"use {data_path}, clear"
    stata.run(load_cmd, quietly=True)

    # margins
    margins_cmd = (
        "margins, at(m_HeatIndex_7d =(10(10)110) vul_index=(-0.25(0.3)0.35) ) atmeans"
    )
    margins_plt_cmd = 'marginsplot, level(83.4) xtitle("Mean Heat Index Lagged 7days") ytitle("Predicted PCPhenoAge Accl")'
    stata.run(margins_cmd, quietly=True)
    stata.run(margins_plt_cmd, quietly=True)

    # save plot
    if data_id is not None:
        save_graph_cmd = f"graph export {fig_dir}/margins_plot_{data_id}.png, replace"
    else:
        num_files = len([f for f in os.listdir(fig_dir) if "margins_plot" in f])
        save_graph_cmd = f"graph export {fig_dir}/margins_plot_{num_files}.png, replace"
    stata.run(save_graph_cmd, quietly=True)  # Save the figure


def draw_shapley_summary_plot(
    model_index: IndexPredictionModel,
    all_interaction_vars_tensor: torch.Tensor,
    interaction_predictor_names: Sequence[str],
    fig_dir: str = os.path.join(TEMP_DIR, "figures"),
    data_id: Union[str, None] = None,
):
    explainer = shap.DeepExplainer(model_index, all_interaction_vars_tensor)
    shap_values = explainer.shap_values(
        all_interaction_vars_tensor[:1000], check_additivity=False
    )
    shap.summary_plot(
        shap_values[:, :],
        all_interaction_vars_tensor[:1000, :].detach().cpu().numpy(),
        feature_names=interaction_predictor_names,
        show=False,
    )
    if data_id is not None:
        fig_name = os.path.join(fig_dir, "shapley_summary_plot_{}.png".format(data_id))
    else:
        num_files = len([f for f in os.listdir(fig_dir) if "shapley_summary_plot" in f])
        fig_name = os.path.join(
            fig_dir, "shapley_summary_plot_{}.png".format(num_files)
        )
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
