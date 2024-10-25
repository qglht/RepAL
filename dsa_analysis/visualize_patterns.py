from curses.ascii import SI
from re import S
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
        "darkgreen"
    )  # 'coolwarm' is a built-in colormap that transitions from blue to red

    for i in range(len(model)):
        fig = plt.figure(dpi=500)
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


def visualize_simple(model: List[np.ndarray], color):
    """Visualize a simulation with shape described below

    Args:
        model (List[np.ndarray]): list of models to plot, of shape time_steps, 3
        title (str): title of the plot
    """
    for i in range(len(model)):
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(projection="3d")
        xyzs = model[i]

        # Loop over the trajectory segments and plot each in black
        for j in range(len(xyzs) - 1):
            start = xyzs[j]
            end = xyzs[j + 1]
            color = color
            ax.plot(
                *zip(start, end), color=color, lw=2
            )  # 'k' is the color code for black

        # Turn off the grid and axis labels
        ax.grid(False)
        ax.axis("off")

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
    ax.grid(False)
    ax.axis("off")
    # Show legend with titles
    plt.legend()

    # Display the plot
    plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

colors = ["#8FBC8F", "#9BC1BC", "#4C1E4F"]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

# Define font sizes and styles for the plot
SIZE_DEFAULT = 12
SIZE_LARGE = 12
plt.rc("font", family="Arial")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels


import matplotlib.pyplot as plt
import numpy as np
import random

import matplotlib.pyplot as plt
import numpy as np
import random


import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import cm


from matplotlib.colors import Normalize
import matplotlib.cm as cm

import matplotlib.lines as mlines


def visualize_separate_plots(
    models: list[np.ndarray], titles: list[str], palette: dict
):
    """Visualize each simulation in a separate 3D plot with mean and sample trials.

    Args:
        models (list[np.ndarray]): list of models to plot, each model is an array of shape (n_time, n_trials, 3)
        titles (list[str]): list of titles for each trajectory
        palette (dict): dictionary mapping each title to a specific color
    """
    # Use the Viridis colormap for coloring
    viridis = cm.get_cmap("viridis", 4)

    # Loop through each model and create a separate plot
    for i in range(len(models)):
        xyzs = models[i]
        title = titles[i]

        # Create a new figure for each plot
        fig = plt.figure(figsize=(8, 6), dpi=500)
        ax = fig.add_subplot(projection="3d")

        # Compute the mean trajectory across trials
        mean_trajectory = np.mean(xyzs, axis=1)
        mean_start_point = mean_trajectory[0, :]  # Starting point of the mean trial

        # Plot 3 randomly sampled trials with Viridis colors
        n_trials = xyzs.shape[1]
        sampled_trials = random.sample(range(n_trials), 4)
        for idx, trial_idx in enumerate(sampled_trials):
            trial = xyzs[:, trial_idx, :]

            # Translate each trial to share the same starting point as the mean trial
            translated_trial = trial - trial[0, :] + mean_start_point

            # Plot the entire trial with a single color from the Viridis palette
            color = viridis(idx / len(sampled_trials))
            ax.plot(
                translated_trial[:, 0],
                translated_trial[:, 1],
                translated_trial[:, 2],
                color=color,
                lw=2.5,
                alpha=0.5,
            )

            # Mark the end of the trial with a colorful triangle
            end_point = translated_trial[-1]
            ax.scatter(
                end_point[0],
                end_point[1],
                end_point[2],
                color=color,  # Triangle is the same color as the trial
                marker="^",
                alpha=1,
                s=100,
            )

        # # Plot the mean trajectory with a thicker line using a stronger viridis color
        # mean_color = viridis(1)
        # ax.plot(
        #     mean_trajectory[:, 0],
        #     mean_trajectory[:, 1],
        #     mean_trajectory[:, 2],
        #     color=mean_color,
        #     lw=3,  # Thicker line for the mean
        # )

        # Mark the start of the mean curve with a black cross
        ax.scatter(
            mean_start_point[0],
            mean_start_point[1],
            mean_start_point[2],
            color="black",
            marker="x",
            s=100,  # Larger marker for the mean
        )

        # # Mark the end of the mean curve with a colorful triangle
        # end_point = mean_trajectory[-1]
        # ax.scatter(
        #     end_point[0],
        #     end_point[1],
        #     end_point[2],
        #     color=mean_color,
        #     marker="^",
        #     s=100,  # Larger marker for the mean
        # )

        # Draw black axes starting from (0,0,0)
        max_range = np.max(
            np.abs(mean_trajectory - mean_start_point)
        )  # Adjust max_range based on the mean trajectory
        ax.quiver(0, 0, 0, max_range, 0, 0, color="black", arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, -max_range, 0, color="black", arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, 0, max_range, color="black", arrow_length_ratio=0.2)

        # Adjust limits
        buffer = 0.1  # Buffer to add around the end points
        ax.set_xlim([-max_range - buffer, max_range + buffer])
        ax.set_ylim([-max_range - buffer, max_range + buffer])
        ax.set_zlim([-max_range - buffer, max_range + buffer])

        # Hide grid and axis if desired
        ax.grid(False)
        ax.axis("off")

        # Create a custom legend for start (cross) and end (triangle)
        cross = mlines.Line2D(
            [], [], color="black", marker="x", markersize=10, lw=0, label="Start"
        )
        triangle = mlines.Line2D(
            [], [], color="black", marker="^", markersize=10, lw=0, label="End"
        )

        # Add the legend with "start" and "end" entries
        ax.legend(handles=[cross, triangle], loc="upper right")

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


