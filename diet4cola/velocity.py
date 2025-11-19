import numpy as np

from typing import Callable

def prior_exp(v_0: float, d: float = 1.0, parameter: float = 0.1, l: float = 100.0) -> float:
    return v_0 * np.exp(-d / l)

def compute_velocity_field(sdf: np.ndarray,
                           velocity_prior: Callable, 
                           v_0: float = 1.0, 
                           k: float = 1.0,
                           falloff_length: float = 100.0) -> np.ndarray:
    width, height = sdf.shape
    data = np.zeros([height, width])
    max_dist = np.max(sdf)

    for y in range(height):
        for x in range(width):
            data[y, x] = np.maximum(0, velocity_prior(v_0, sdf[y, x], k, falloff_length))
    return data

def compute_velocity_fields(sdfs: np.ndarray,
                            velocity_prior: Callable, 
                            timepoints: np.array, 
                            initial_velocities: np.array, 
                            falloff_length: float,
                            count: int,) -> np.ndarray:
    if count != len(sdfs):
        raise ValueError(f'Expected {count} SDFs but got {sdfs.shape[0]}')
    if count != len(timepoints):
        raise ValueError(f'Expected {count} timepoints but got {len(timepoints)}')
    if count != len(initial_velocities):
        raise ValueError(f'Expected {count} initial velocities but got {len(initial_velocities)}')

    velocity_fields = []
    for i in range(count):
        velocity_fields.append(compute_velocity_field(sdfs[i], velocity_prior, initial_velocities[i], falloff_length))
    return np.array(velocity_fields)