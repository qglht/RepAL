import matplotlib.pyplot as plt
import numpy as np
import ipdb
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import CubicSpline, Rbf

# Set seed for reproducibility
# np.random.seed(42)

from scipy.signal import savgol_filter

def savitzky_golay_smooth(data, window_size=5, poly_order=2):
    # Ensure window size is odd and larger than poly_order
    if window_size % 2 == 0:
        window_size += 1
    return np.array([savgol_filter(data[:, :, i], window_size, poly_order) for i in range(data.shape[2])]).transpose(1, 2, 0)




def normalize_within_unit_volume(array):
    # Find the minimum and maximum values in the entire 3D array
    min_value = np.min(array)
    max_value = np.max(array)

    # Calculate scaling factor to fit the entire array within the unit volume
    scale_factor = 1.0 / (max_value - min_value+1e-6)

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

def generate_initial_points(num_samples=100):
    """Generate initial points."""
    # Generate 100 points where each coordinate x, y, z lies within the range [0, 1)
    points = np.random.rand(num_samples, 3)
    return points

def generate_random_direction(direction):
    """Generate a random direction."""
    # Generate a random direction by adding a small random noise to the original direction
    noise = np.random.normal(0,1)
    axis_flipped = np.random.randint(1,3)
    for ax in range(3):
        if ax != axis_flipped:
            direction[ax] = direction[ax]*noise
        else : 
            direction[ax] = -direction[ax]*noise
    return noise*direction

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
    t_values = np.empty((num_steps,3))
    t_values[0] = initial_point
    for i in range(num_steps-1):
        t_values[i + 1] = t_values[i] + i*direction
    t_values = normalize_within_unit_volume(t_values)
    # smooth_noise = np.random.normal(loc=0, scale=0.01, size=(num_steps,3))
    # t_values = t_values + smooth_noise
    return t_values

class Simulation:
    def __init__(self, num_samples=100, num_steps=10000, dt=0.01):
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.dt = dt
        self.type = type
        self.initial_points = generate_initial_points(self.num_samples)
        self.simulation = np.empty((self.num_samples, self.num_steps, 3))
        self.perturbations_record = {}

    def lorenz_simulation(self, initial_point, **lorenz_params):

        xyzs = np.empty((self.num_steps, 3))  # Need one more for the initial values
        xyzs[0] = initial_point  # Set initial values
        for i in range(self.num_steps-1):
            xyzs[i + 1] = xyzs[i] + lorenz_step(xyzs[i], **lorenz_params) * self.dt 
        xyzs = normalize_within_unit_volume(xyzs)
        return xyzs

    def simulation_lorenz(self, lorenz_params):
        for sample in range(self.num_samples):
            self.simulation[sample] = self.lorenz_simulation(self.initial_points[sample], **lorenz_params)

    def simulation_line(self):
        direction= np.array([1,1,1])
        for sample in range(self.num_samples):
            new_direction = generate_random_direction(direction)
            self.simulation[sample] = generate_3d_line(self.initial_points[sample], new_direction, num_steps=self.num_steps)

    def perturbation(self, perturbation_scale: float = 0.02, model: str = 'model1', epoch: str = 'epoch1'):
        perturbed_curve = self.simulation.copy()
        noise = np.random.normal(0, 1, size=(self.num_samples, self.num_steps, 3))
        perturbed_curve = np.sqrt(1-perturbation_scale**2)*perturbed_curve + perturbation_scale*noise
        # smoothed_curve = savitzky_golay_smooth(perturbed_curve, window_size=20, poly_order=5)
        try:
            self.perturbations_record[model][epoch] = perturbed_curve
        except KeyError:
            self.perturbations_record[model] = {}
            self.perturbations_record[model][epoch] = perturbed_curve