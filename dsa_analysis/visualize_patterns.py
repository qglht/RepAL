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

