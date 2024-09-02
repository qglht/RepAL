import matplotlib.pyplot as plt
from typing import List
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap


def visualize(model: List[np.ndarray], title: str):
    """Visualize a simulation with shape described below

    Args:
        model (List[np.ndarray]): list of models to plot, of shape time_steps, 3
        title (str): title of the plot
    """
    # Define a custom colormap that transitions from blue to red
    cmap = get_cmap(
        "coolwarm"
    )  # 'coolwarm' is a built-in colormap that transitions from blue to red

    for i in range(len(model)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        xyzs = model[i]

        # Use a colormap to color the trajectory over time
        # Normalize time steps to the range [0, 1]
        norm = Normalize(vmin=0, vmax=len(xyzs) - 1)

        # Loop over the trajectory segments and plot each with a color gradient
        for j in range(len(xyzs) - 1):
            start = xyzs[j]
            end = xyzs[j + 1]
            color = cmap(norm(j))
            ax.plot(*zip(start, end), color=color, lw=2)

        # Set labels and title
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(title)

        plt.show()


import matplotlib.pyplot as plt
from typing import List
import numpy as np


def visualize_same_plot(model: List[np.ndarray], titles: List[str], palette):
    """Visualize multiple simulations in a single 3D plot with different colors and markers for each trajectory.

    Args:
        model (List[np.ndarray]): list of models to plot, where each model is an array of shape (time_steps, 3)
        titles (List[str]): list of titles for each trajectory for the legend
    """
    # List of distinct colors for each curve
    colors = ["red", "blue", "green", "purple", "orange", "grey", "brown"]

    # Create a single 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Initialize limits for zooming
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    # Plot each trajectory on the same plot
    for i in range(len(model)):
        xyzs = model[i]
        title = titles[i]
        color = palette.get(title, "gray")  # Default to black if title not in palette

        # Plot the trajectory using the constant color and add to the legend
        ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], color=color, lw=2, label=title)

        # Mark the beginning of the curve with a larger triangle
        ax.scatter(
            xyzs[0, 0],
            xyzs[0, 1],
            xyzs[0, 2],
            color=color,
            marker="^",
            s=150,
            edgecolor="black",
        )

        # Mark the end of the curve with a larger cross
        ax.scatter(
            xyzs[-1, 0],
            xyzs[-1, 1],
            xyzs[-1, 2],
            color=color,
            marker="x",
            s=150,
            edgecolor="black",
        )

        # Update limits for zooming
        x_min = min(x_min, xyzs[:, 0].min())
        x_max = max(x_max, xyzs[:, 0].max())
        y_min = min(y_min, xyzs[:, 1].min())
        y_max = max(y_max, xyzs[:, 1].max())
        z_min = min(z_min, xyzs[:, 2].min())
        z_max = max(z_max, xyzs[:, 2].max())

    # Set labels and title
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    # Adjust limits to zoom in around the end points
    buffer = 0.1  # Buffer to add around the end points
    ax.set_xlim([x_min - buffer, x_max + buffer])
    ax.set_ylim([y_min - buffer, y_max + buffer])
    ax.set_zlim([z_min - buffer, z_max + buffer])

    # Show legend with titles
    plt.legend()

    # Display the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import List


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import List


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import List, Dict


# Function to plot multiple trajectories on a single axis with custom settings and colors
def visualize_on_axis(
    ax,
    model: List[np.ndarray],
    titles: List[str],
    palette: Dict[str, str],
    fontsize: int,
    ax_labels: List[str],
):
    """
    Visualize multiple simulations on a given 3D axis with different colors from a provided palette.

    Args:
        ax: Matplotlib Axes3D object where the plot will be drawn.
        model (List[np.ndarray]): list of models to plot, where each model is an array of shape (time_steps, 3)
        titles (List[str]): list of titles for each trajectory for the legend
        palette (Dict[str, str]): Dictionary mapping each title to a specific color.
        fontsize (int): Font size for the axis labels and title.
        ax_labels (List[str]): List of axis labels.
    """
    # Initialize limits for zooming
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    # Plot each trajectory on the same plot
    for i in range(len(model)):
        xyzs = model[i]
        title = titles[i]
        color = palette.get(title, "gray")  # Default to black if title not in palette

        # Plot the trajectory using the constant color and add to the legend
        ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], color=color, lw=2, label=title)

        # Mark the beginning of the curve with a larger triangle
        ax.scatter(
            xyzs[0, 0],
            xyzs[0, 1],
            xyzs[0, 2],
            color=color,
            marker="^",
            s=150,
            edgecolor="black",
        )

        # Mark the end of the curve with a larger cross
        ax.scatter(
            xyzs[-1, 0],
            xyzs[-1, 1],
            xyzs[-1, 2],
            color=color,
            marker="x",
            s=150,
            edgecolor="black",
        )

        # Update limits for zooming
        x_min = min(x_min, xyzs[:, 0].min())
        x_max = max(x_max, xyzs[:, 0].max())
        y_min = min(y_min, xyzs[:, 1].min())
        y_max = max(y_max, xyzs[:, 1].max())
        z_min = min(z_min, xyzs[:, 2].min())
        z_max = max(z_max, xyzs[:, 2].max())

    # Set labels and title with custom font size
    ax.set_xlabel(ax_labels[0], fontsize=fontsize)
    ax.set_ylabel(ax_labels[1], fontsize=fontsize)
    ax.set_zlabel(ax_labels[2], fontsize=fontsize)

    # Adjust limits to zoom in around the end points
    buffer = 0.1  # Buffer to add around the end points
    ax.set_xlim([x_min - buffer, x_max + buffer])
    ax.set_ylim([y_min - buffer, y_max + buffer])
    ax.set_zlim([z_min - buffer, z_max + buffer])

    # Show legend with titles
    ax.legend(fontsize=fontsize)


def plot_with_gridspec(
    model1: List[np.ndarray],
    titles1: List[str],
    model2: List[np.ndarray],
    titles2: List[str],
    palette1: Dict[str, str],
    palette2: Dict[str, str],
    ax_labels: List[str],
    fig_size: int,
    fontsize: int,
    plot1: str,
    plot2: str,
):
    """
    Plot two sets of models on the same figure using GridSpec with customized font, plot settings, and color palettes.

    Args:
        model1 (List[np.ndarray]): list of models for the first plot
        titles1 (List[str]): list of titles for the first plot's trajectories
        model2 (List[np.ndarray]): list of models for the second plot
        titles2 (List[str]): list of titles for the second plot's trajectories
        palette1 (Dict[str, str]): Color palette for the first subplot.
        palette2 (Dict[str, str]): Color palette for the second subplot.
        ax_labels (List[str]): List of axis labels.
        fig_size (int): Size of the figure.
        fontsize (int): Font size for the axis labels and title.
    """
    # Create a new figure with specified size
    fig = plt.figure(figsize=(fig_size * 2, fig_size))

    # Set up GridSpec layout
    gs = GridSpec(1, 2, figure=fig)

    # First subplot with palette1
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    visualize_on_axis(ax1, model1, titles1, palette1, fontsize, ax_labels)
    ax1.set_title(plot1, fontsize=fontsize)

    # Second subplot with palette2
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    visualize_on_axis(ax2, model2, titles2, palette2, fontsize, ax_labels)
    ax2.set_title(plot2, fontsize=fontsize)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the combined plot
    plt.show()
