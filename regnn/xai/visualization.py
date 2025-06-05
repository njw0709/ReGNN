import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(
        range(data.shape[1]),
        labels=col_labels,
        rotation=-30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    annotations_np,
    textcolors=("black", "white"),
    threshold=None,
    textsize=10,
    padding=0.1,
    **textkw,
):
    """
    A function to annotate a heatmap with dynamically sized boxes.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    annotations
        Data used to annotate.  If None, the image's data is used.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    textsize
        Base font size for the annotations. The actual size will be adjusted
        based on text content. Optional.
    padding
        Padding around text as a fraction of the cell size. Optional.
    **textkw
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the figure and axes
    fig = im.axes.figure
    ax = im.axes

    # Calculate the maximum text length in each cell
    max_text_length = max(len(str(text)) for row in annotations_np for text in row)

    # Adjust figure size based on text content
    current_size = fig.get_size_inches()
    # Scale width based on text length, but keep aspect ratio
    new_width = current_size[0] * (1 + padding * max_text_length / 10)
    fig.set_size_inches(new_width, current_size[1])

    # Loop over the data and create a `Text` for each "pixel".
    texts = []
    for i in range(annotations_np.shape[0]):
        for j in range(annotations_np.shape[1]):
            # Calculate dynamic text size based on content length
            cell_text = str(annotations_np[i, j])
            dynamic_textsize = textsize * (1 - 0.1 * len(cell_text) / max_text_length)

            # Create text with dynamic size
            text = ax.text(j, i, cell_text, fontsize=dynamic_textsize, **kw)
            texts.append(text)

    # Use tight_layout to automatically adjust spacing
    # fig.tight_layout()

    return texts
