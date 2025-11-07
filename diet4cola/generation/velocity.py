import numpy as np

from tqdm import tqdm
from typing import Callable

def prior_exponential(t: float, v_0: float = 1.0, k: float = 1.0, d: float = 1.0) -> float:
    return v_0 * np.exp(-(t * d) / k)

def prior_linear(t: float, v_0: float = 1.0, k: float = 1.0, d: float = 1.0) -> float:
    return np.maximum(0, v_0 - k * t * d)

def prior_hyperbolic(t: float, v_0: float = 1.0, k: float = 1.0, d: float = 1.0) -> float:
    t = t + 1e-8
    return v_0 / (1 + k * t * d)

def prior_power(t: float, v_0: float = 1.0, k: float = 1.0, d: float = 1.0) -> float:
    return v_0 * (1 + t) ** (-k * d)

def compute_velocity_field(sdf: np.ndarray,
                           velocity_prior: Callable, 
                           t: float, 
                           v_0: float = 1.0, 
                           k: float = 1.0) -> np.ndarray:
    width, height = sdf.shape
    data = np.zeros([height, width])

    for y in range(height):
        for x in range(width):
            data[y, x] = velocity_prior(t, v_0, k, sdf[y, x])
    return data

def compute_velocity_field_multi(sdfs: np.ndarray,
                                 velocity_prior: Callable, 
                                 timepoints: np.array, 
                                 initial_velocities: np.array, 
                                 parameters: np.array,
                                 count: int,) -> np.ndarray:
    if count != len(sdfs):
        raise ValueError(f'Expected {count} SDFs but got {sdfs.shape[0]}')
    if count != len(timepoints):
        raise ValueError(f'Expected {count} timepoints but got {len(timepoints)}')
    if count != len(initial_velocities):
        raise ValueError(f'Expected {count} initial velocities but got {len(initial_velocities)}')
    if count != len(parameters):
        raise ValueError(f'Expected {count} paramteres but got {len(parameters)}')

    velocity_fields = []
    for i in tqdm(range(count)):
        velocity_fields.append(compute_velocity_field(sdfs[i], velocity_prior, timepoints[i], initial_velocities[i], parameters[i]))
    return np.array(velocity_fields)