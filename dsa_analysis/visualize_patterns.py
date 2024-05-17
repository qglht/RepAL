import matplotlib.pyplot as plt
from typing import List
import numpy as np


def visualize(model: List[np.ndarray], title):
    """Visualize a simulation with shape described below

    Args:
        model (List[np.ndarray]): list of models to plot, of shape time_steps, 3
    """
    for i in range(len(model)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        xyzs = model[i]
        to_plot = xyzs
        # Plot trajectory of first
        ax.plot(*to_plot.T, lw=0.5)

        # ax.scatter(*fourth, color='black')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(title)

        plt.show()
