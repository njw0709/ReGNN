from mihm.model.mihm import IndexPredictionModel, MIHM
from mihm.model.modelutils import get_index_prediction_weights
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
    stata_setup.config("/usr/local/stata17", "se")


def init_shap():
    shap.initjs()


def evaluate_significance(
    df_orig: pd.DataFrame,
    outcome: str,
    index_predictions: np.ndarray,
):
    # save to file
    df_orig["vul_index"] = index_predictions

    formula_str = """
    {} ~ m_HeatIndex_7d + m_HeatIndex_7d:vul_index + pmono
    + PNK_pct + PBcell_pct + PCD8_Plus_pct + PCD4_Plus_pct + PNCD8_Plus_pct 
    + age2016 + C(female) + C(racethn) + eduy + ihs_wealthf2016 + C(smoke2016) + C(drink2016) 
    + bmi2016 + tractdis + C(urban) + C(mar_cat2) + C(psyche2016) + C(stroke2016) + C(hibpe2016) + C(diabe2016) 
    + C(hearte2016) + dep2016 + ltactx2016 + mdactx2016 + vgactx2016 + C(living2016) + C(division) + adl2016
    """.format(
        outcome
    )
    formula_str = formula_str.replace("\n", "")

    model = smf.ols(formula=formula_str, data=df_orig).fit()
    summary = model.summary()
    # get significance of interaction
    coeffs = summary.tables[1]
    col_names = coeffs.data[0]
    var_names = [row[0] for row in coeffs.data[1:]]
    interaction_idx = var_names.index("m_HeatIndex_7d:vul_index")
    pval = float(coeffs.data[interaction_idx + 1][4])
    return pval


def compute_index_prediction(
    model: MIHM, interaction_predictors: torch.Tensor
) -> np.ndarray:

    index_model = model.get_index_prediction_model()
    index_model.to(interaction_predictors.device).eval()

    index_prediction = index_model(interaction_predictors).detach().cpu().numpy()

    return index_prediction


def evaluate_significance_stata(
    df_orig: pd.DataFrame,
    outcome: str,
    index_predictions: np.ndarray,
    save_dir: str = os.path.join(TEMP_DIR, "data"),
    id: Union[str, None] = None,
):
    init_stata()
    from pystata import stata

    if id is not None:
        save_path = os.path.join(save_dir, f"index_prediction_{id}.dta")
    else:
        num_files = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f"index_prediction_{num_files}.dta")

    # save to file
    df_orig["vul_index"] = index_predictions
    df_orig.to_stata(save_path, write_index=False)

    # run regression and get significance
    load_cmd = f"use {save_path}, clear"
    stata.run(load_cmd, quietly=True)
    regress_cmd = "regress {} m_HeatIndex_7d c.m_HeatIndex_7d#c.vul_index pmono PNK_pct PBcell_pct PCD8_Plus_pct PCD4_Plus_pct PNCD8_Plus_pct age2016 i.female i.racethn eduy ihs_wealthf2016 i.smoke2016 i.drink2016 bmi2016 tractdis i.urban i.mar_cat2 i.psyche2016 i.stroke2016 i.hibpe2016 i.diabe2016 i.hearte2016 dep2016 ltactx2016 mdactx2016 vgactx2016 i.living2016 i.division adl2016".format(
        outcome
    )
    stata.run(regress_cmd, quietly=True)
    regression_results = stata.get_return()

    interaction_pval = regression_results["r(table)"][3, 1]

    # vif
    stata.run("vif", quietly=True)
    vif_results = stata.get_return()
    vif_heat = vif_results["r(vif_1)"]
    vif_inter = vif_results["r(vif_2)"]

    return interaction_pval, (vif_heat, vif_inter)


def draw_margins_plot(
    data_path: str,
    fig_dir: str = os.path.join(TEMP_DIR, "figures"),
    id: Union[str, None] = None,
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
    if id is not None:
        save_graph_cmd = f"graph export {fig_dir}/margins_plot_{id}.png, replace"
    else:
        num_files = len([f for f in os.listdir(fig_dir) if "margins_plot" in f])
        save_graph_cmd = f"graph export {fig_dir}/margins_plot_{num_files}.png, replace"
    stata.run(save_graph_cmd, quietly=True)  # Save the figure


def draw_shapley_summary_plot(
    model_index: IndexPredictionModel,
    all_interaction_vars_tensor: torch.Tensor,
    interaction_predictor_names: Sequence[str],
    fig_dir: str = os.path.join(TEMP_DIR, "figures"),
    id: Union[str, None] = None,
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
    if id is not None:
        fig_name = os.path.join(fig_dir, "shapley_summary_plot_{}.png".format(id))
    else:
        num_files = len([f for f in os.listdir(fig_dir) if "shapley_summary_plot" in f])
        fig_name = os.path.join(
            fig_dir, "shapley_summary_plot_{}.png".format(num_files)
        )
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
