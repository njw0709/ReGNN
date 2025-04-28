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
