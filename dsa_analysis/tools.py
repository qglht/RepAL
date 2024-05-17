import yaml
import numpy as np


def load_config(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


# some code that's relevant for processing our data
def flatten_x(x1):
    # this will flatten the first 2 dimensions (conditions, trials) to 1 dimension
    return x1.reshape(x1.shape[0] * x1.shape[1], x1.shape[2], x1.shape[3])


def normalize_within_unit_volume(array):
    # Find the minimum and maximum values in the entire 3D array
    min_value = np.min(array)
    max_value = np.max(array)

    # Calculate scaling factor to fit the entire array within the unit volume
    scale_factor = 1.0 / (max_value - min_value)

    # Normalize the array
    normalized_array = (array - min_value) * scale_factor

    return normalized_array
