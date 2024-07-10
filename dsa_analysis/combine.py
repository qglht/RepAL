import matplotlib.pyplot as plt
import numpy as np
import ipdb
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import copy


def downsample(array, factor):
    return array[::factor, :]


def normalize_within_unit_volume(array):
    # Find the minimum and maximum values in the entire 3D array
    copy_array = copy.deepcopy(array)
    min_value = np.min(copy_array)
    max_value = np.max(copy_array)

    # Calculate scaling factor to fit the entire array within the unit volume
    scale_factor = 1.0 / (max_value - min_value)

    # Normalize the array
    normalized_array = (copy_array - min_value) * scale_factor

    return normalized_array


def attach_2_motifs(motifA, motifB):
    """Combine two motifs by concatenating them along the time axis.

    Args:
    - motifA: A numpy array of shape (num_steps, 3) representing the first motif.
    - motifB: A numpy array of shape (num_steps, 3) representing the second motif.

    Returns:
    - A numpy array of shape (num_steps*2, 3) representing the combined motif.
    """
    new_motif = np.empty((motifA.shape[0] + motifB.shape[0], motifA.shape[1]))
    new_motif[: motifA.shape[0]] = motifA
    new_motif[motifA.shape[0]] = new_motif[motifA.shape[0] - 1] + 0.0001 * (
        motifA[-1] - motifA[-2]
    )
    for i in range(1, motifB.shape[0]):
        new_motif[motifA.shape[0] + i] = (
            new_motif[motifA.shape[0] + i - 1] + motifB[i] - motifB[i - 1]
        )
    return new_motif


def combine_2_motifs(motifA, motifB, mixing_level=0):
    """Combine two motifs by concatenating them along the time axis.

    Args:
    - motifA: A numpy array of shape (num_steps, 3) representing the first motif.
    - motifB: A numpy array of shape (num_steps, 3) representing the second motif.

    Returns:
    - A numpy array of shape (num_steps*2, 3) representing the combined motif.
    """
    new_motif = np.empty((motifA.shape[0], motifA.shape[1]))
    for i in range(motifB.shape[0]):
        new_motif[i] = (1 - mixing_level) * motifB[i] + mixing_level * (
            motifA[i] - motifA[0]
        )
    return new_motif


def combine_motifs(motifs: List[np.ndarray], method="attach", mixing_level=0):
    """Combine multiple motifs by concatenating them along the time axis.

    Args:
    - motifs: A numpy array of shape (num_motifs, num_steps, 3) representing the motifs.

    Returns:
    - A numpy array of shape (num_motifs*num_steps, 3) representing the combined motifs.
    """
    if method == "attach":
        new_motif = attach_2_motifs(motifs[0], motifs[1])
        for i in range(2, len(motifs)):
            new_motif = attach_2_motifs(new_motif, motifs[i])
    else:
        new_motif = combine_2_motifs(motifs[0], motifs[1], mixing_level=mixing_level)
        for i in range(2, len(motifs)):
            new_motif = combine_2_motifs(
                new_motif, motifs[i], mixing_level=mixing_level
            )
    return new_motif


def combine_simulations(simulations: List[np.ndarray], method="attach", mixing_level=0):
    combined_simulation = np.empty(
        (simulations[0].shape[0], simulations[0].shape[1], simulations[0].shape[2])
    )
    for i in range(simulations[0].shape[0]):
        combine_simulation_unormalized = combine_motifs(
            [simulation[i] for simulation in simulations],
            method=method,
            mixing_level=mixing_level,
        )
        combined_simulation[i] = (
            downsample(combine_simulation_unormalized, len(simulations))
            if method == "attach"
            else combine_simulation_unormalized
        )
        combined_simulation[i] = (
            normalize_within_unit_volume(combined_simulation[i])
            if method == "attach"
            else combined_simulation[i]
        )
    return combined_simulation
