import matplotlib.pyplot as plt
import numpy as np
import ipdb
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import CubicSpline, Rbf


# Set seed for reproducibility
np.random.seed(42)


def normalize_within_unit_volume(array):
    # Find the minimum and maximum values in the entire 3D array
    min_value = np.min(array)
    max_value = np.max(array)

    # Calculate scaling factor to fit the entire array within the unit volume
    scale_factor = 1.0 / (max_value - min_value)

    # Normalize the array
    normalized_array = (array - min_value) * scale_factor

    return normalized_array


def lorenz_step(xyz, *, s=10, r=30, b=2.667):
    """Perform one step of the Lorenz system."""
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


# def perturb_curve(curve, perturbation_scale: float = 0.1):
#     perturbed_curve = curve.copy()
#     num_points = curve.shape[0]
#     perturbation = np.random.normal(0, perturbation_scale, size=(num_points, 3))
#     perturbed_curve += perturbation
#     return perturbed_curve


def perturb_curve_diffusion(curve, perturbation_scale: float = 0.02):
    perturbed_curve = curve.copy()
    num_points = curve.shape[0]
    perturbation = np.random.normal(0, 1, size=(num_points, 3))
    perturbed_curve = (
        np.sqrt(1 - perturbation_scale**2) * perturbed_curve
        + perturbation_scale * perturbation
    )
    return perturbed_curve


def smooth_curve(curve, sigma: float = 2.0):
    smoothed_curve = np.zeros_like(curve)
    for i in range(curve.shape[1]):  # Iterate over dimensions
        smoothed_curve[:, i] = gaussian_filter1d(curve[:, i], sigma=sigma)
    return smoothed_curve


def lorenz_simulation(
    dt=0.01,
    num_steps=10000,
    initial_point=(0.0, 1.0, 1.05),
    perturbation_scale: float = 0.02,
    sigma: float = 2.0,
    order: int = 0,
    **lorenz_params
):
    """Simulate the Lorenz system."""
    xyzs = np.empty((num_steps, 3))  # Need one more for the initial values
    xyzs[0] = initial_point  # Set initial values
    for i in range(num_steps - 1):
        xyzs[i + 1] = xyzs[i] + lorenz_step(xyzs[i], **lorenz_params) * dt
    xyzs = normalize_within_unit_volume(xyzs)
    if order > 0:
        for i in range(order):
            xyzs = perturb_curve_diffusion(xyzs, perturbation_scale)
    # xyzs = smooth_curve(xyzs, sigma)
    xyzs = normalize_within_unit_volume(xyzs)
    # smooth_noise = np.random.normal(loc=0, scale=0.01, size=(num_steps,3))
    # xyzs = xyzs + smooth_noise
    return xyzs


def generate_3d_line(initial_point, direction, num_steps=100):
    """
    Generate points along a line segment in 3D space.

    Args:
    - point1: Coordinates of the starting point of the line.
    - direction: Direction vector of the line.
    - length: Length of the line segment.

    Returns:
    - An array of shape (num_points, 3) containing the coordinates of the points along the line.
    """
    t_values = np.empty((num_steps, 3))
    t_values[0] = initial_point
    for i in range(num_steps - 1):
        t_values[i + 1] = t_values[i] + i * direction
    t_values = normalize_within_unit_volume(t_values)
    # smooth_noise = np.random.normal(loc=0, scale=0.01, size=(num_steps,3))
    # t_values = t_values + smooth_noise
    return t_values


def generate_initial_points(num_samples=100):
    """Generate initial points."""
    # Generate 100 points where each coordinate x, y, z lies within the range [0, 1)
    points = np.random.rand(num_samples, 3)
    return points


def generate_random_direction(direction):
    """Generate a random direction."""
    # Generate a random direction by adding a small random noise to the original direction
    noise = np.random.normal(0, 1)
    axis_flipped = np.random.randint(1, 3)
    for ax in range(3):
        if ax != axis_flipped:
            direction[ax] = direction[ax] * noise
        else:
            direction[ax] = -direction[ax] * noise
    return noise * direction


def simulation(
    dt,
    lorenz_parameters,
    num_samples,
    num_steps,
    perturbation: bool = False,
    perturbation_scale: float = 0.1,
    smoothing: bool = False,
    sigma: float = 2.0,
):
    """Run simulations for different Lorenz parameter conditions."""
    conditions = len(lorenz_parameters)
    sub_conditions = len(lorenz_parameters["one_attractor"])
    simulations = np.empty((conditions, sub_conditions, num_samples, num_steps, 3))
    initial_points = generate_initial_points(num_samples)
    for i, (_, params) in enumerate(lorenz_parameters.items()):
        condition = i
        for param in params:
            sub_condition = params.index(param)
            for sample in range(num_samples):
                simulations[condition, sub_condition, sample] = lorenz_simulation(
                    dt,
                    num_steps,
                    initial_point=initial_points[sample],
                    perturbation=perturbation,
                    perturbation_scale=perturbation_scale,
                    smoothing=smoothing,
                    sigma=sigma,
                    **param
                )
    return simulations


def simulation_line(num_steps, num_samples):
    simulations = np.empty((num_samples, num_steps, 3))
    initial_points = generate_initial_points(num_samples)
    direction = np.array([0.0001, 0.0001, 0.0001])
    for sample in range(num_samples):
        new_direction = generate_random_direction(direction)
        simulations[sample] = generate_3d_line(
            initial_points[sample], new_direction, num_steps=num_steps
        )
    return simulations


def simulation_lorenz(
    dt,
    lorenz_parameters,
    num_samples,
    num_steps,
    perturbation_scale: float = 0.02,
    order: int = 0,
    sigma: float = 2.0,
):
    """Run simulations for different Lorenz parameter conditions."""
    simulations = np.empty((num_samples, num_steps, 3))
    initial_points = generate_initial_points(num_samples)
    param = lorenz_parameters
    for sample in range(num_samples):
        simulations[sample] = lorenz_simulation(
            dt,
            num_steps,
            initial_point=initial_points[sample],
            perturbation_scale=perturbation_scale,
            sigma=sigma,
            order=order,
            **param
        )
    return simulations
