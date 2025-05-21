from .utils import init_stata


def draw_margins_plot_stata(
    margins_cmd: str,
    margins_plot_format_cmd: str,
    save_fig: bool = True,
    fig_save_path: str = "margins_plot.png",
):
    stata = init_stata()
    stata.run(margins_cmd, quietly=True)
    stata.run(margins_plot_format_cmd, quietly=True)

    # save plot
    if save_fig:
        save_graph_cmd = f"graph export {fig_save_path}, replace"
    stata.run(save_graph_cmd, quietly=True)  # Save the figure
