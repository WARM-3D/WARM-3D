import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class PlotUtils:
    def __init__(self):
        pass

    @classmethod
    def plot_histogram(
            self,
            ax,
            all_values,
            num_bins,
            range_list,
            x_label,
            y_label,
            bin_labels,
            use_log_scale=False,
            y_max=None,
            step_size=None,
            color_bar_labels=None,
    ):

        ax.set_xticks(bin_labels)
        ax.set_xlim(left=0)
        if use_log_scale:
            ax.set_yscale("log")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        n, bins, patches = plt.hist(all_values, bins=num_bins, range=range_list, edgecolor="black")
        y_max = np.max(n) if y_max is None else y_max
        # round to next 50
        y_max = int(np.ceil(y_max / 50.0)) * 50
        # To normalize your values
        col = (n - n.min()) / (y_max - n.min())
        cm = plt.cm.get_cmap("RdYlBu_r")
        # Get the colormap colors
        my_cmap = cm(np.arange(cm.N))
        # Set alpha
        my_cmap[:, -1] = np.linspace(0.5, 0.5, cm.N)
        # Create new colormap
        my_cmap = ListedColormap(my_cmap)
        patch_idx = 0
        for c, p in zip(col, patches):
            color = cm(c)
            color = (color[0], color[1], color[2], 0.5)
            plt.setp(p, "facecolor", color)
            patch_idx += 1
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        # add legend with color map
        Z = [[0, 0], [0, 0]]
        # levels = range(0, y_max, step_size)
        # generate 7 levels between 0 and 7000
        step_size = int(y_max / 50) if step_size is None else step_size

        # np.linspace(2.0, 3.0, num=5)
        # array([2., 2.25, 2.5, 2.75, 3.])
        levels = np.linspace(0, y_max, 11)

        # convert levels to strings
        contour = plt.contourf(Z, levels, cmap=my_cmap)
        color_bar = plt.colorbar(contour)
        color_bar.ax.tick_params(labelsize=14)
        if color_bar_labels is not None:
            color_bar.ax.set_yticklabels(color_bar_labels)

    @classmethod
    def get_class_colors(self, alpha):
        class_colors = [
            (0, 0.8, 0.96, alpha),
            (0.25, 0.91, 0.72, alpha),
            (0.35, 1, 0.49, alpha),
            (0.92, 0.81, 0.21, alpha),
            (0.72, 0.64, 0.33, alpha),
            (0.85, 0.54, 0.52, alpha),
            (0.91, 0.46, 0.97, alpha),
            (0.69, 0.55, 1, alpha),
            (0.4, 0.42, 0.98, alpha),
            (0.78, 0.78, 0.78, alpha),
        ]
        return class_colors
