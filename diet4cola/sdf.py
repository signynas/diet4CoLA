import numpy as np

from numpy import array, clip, dot, stack
from numpy.linalg import norm
from tqdm import tqdm
from typing import Callable

from concurrent.futures import ProcessPoolExecutor

def sdf_segment(P: np.ndarray,
                A: np.ndarray,
                B: np.ndarray,
                t: float) -> float:
    PA = P - A
    BA = B - A
    h = clip(dot(PA, BA) / dot(BA, BA), 0.0, 1.0)
    return norm(PA - BA * h)

def sdf_box(P: np.ndarray,
            A: np.ndarray,
            B: np.ndarray,
            t: float) -> float:
    half_th = t / 2.0

    BA = B - A
    length = norm(BA)
    dir_vec = BA / length
    perp = array([-dir_vec[1], dir_vec[0]])
    PA = P - A

    # Local frame coordinates
    x = dot(PA, dir_vec)
    y = dot(PA, perp)

    hx, hy = length / 2.0, half_th
    q = stack([x - hx, y], axis=-1)

    # Compute distance to box
    d = np.abs(q) - array([hx, hy])
    outside = np.maximum(d, 0)
    dist_out = np.linalg.norm(outside, axis=-1)
    inside = np.minimum(np.maximum(d[..., 0], d[..., 1]), 0)

    return dist_out + inside

def sdf_capsule(P: np.ndarray, 
                A: np.ndarray, 
                B: np.ndarray, 
                w: float) -> float:
    AB = B - A
    t = clip(dot(P - A, AB) / dot(AB, AB), 0, 1)
    closest = A + t * AB
    return norm(P - closest) - w

def sdf_vesica(P: np.ndarray,
               A: np.ndarray,
               B: np.ndarray,
               w: float) -> float:
    r = 0.5 * norm(B - A)
    d = 0.5 * (r * r - w * w) / w
    v = (B - A) / r
    C = 0.5 * (B + A)
    
    # 2x2 rotation matrix equivalent in numpy
    mat = array([[v[1], v[0]],
                [-v[0], v[1]]])
    
    Q = 0.5 * np.abs(mat @ (P - C))
    
    if r * Q[0] < d * (Q[1] - r):
        H = array([0.0, r, 0.0])
    else:
        H = array([-d, 0.0, d + w])
    
    return norm(Q - H[:2]) - H[2]

def round(P: np.ndarray,
          A: np.ndarray,
          B: np.ndarray,
          w: float,
          r: float,
          sdf: Callable) -> float:
    return sdf(P, A, B, w) - r

def compute_sdf(width: int,
                height: int,
                origin: tuple[int, int],
                destination: tuple[int, int],
                parameter: float,
                clip_zero: bool,
                sdf: Callable) -> np.ndarray:
    A       = np.array([origin[0], origin[1]])
    B       = np.array([destination[0], destination[1]])
    data    = np.zeros([height, width])
    
    for y in range(height):
        for x in range(width):
            P = np.array([x, y])
            data[y, x] = sdf(P, A, B, parameter)
    if clip_zero:
        data = np.where(data < 0, 0, data)
    return data

def compute_sdf_multi(width: int,
                      height: int,
                      origin: tuple[int, int],
                      destination: tuple[int, int],
                      parameters: np.array,
                      clip_zero: bool,
                      sdf: Callable, 
                      count: int) -> np.ndarray:
    if count != len(parameters):
        raise ValueError(f'Expected {count} paramteres but got {len(parameters)}')
    sdfs = []
    for i in tqdm(range(count)): 
        sdfs.append(compute_sdf(width, height, origin, destination, parameters[i], clip_zero, sdf))
    return np.array(sdfs)

def compute_rounded_sdf(width: int,
                        height: int,
                        origin: tuple[int, int],
                        destination: tuple[int, int],
                        parameter: float,
                        radius: float,
                        clip_zero: bool,
                        sdf: Callable) -> np.ndarray:
    A       = np.array([origin[0], origin[1]])
    B       = np.array([destination[0], destination[1]])
    data    = np.zeros([height, width])
    
    for y in range(height):
        for x in range(width):
            P = np.array([x, y])
            data[y, x] = round(P, A, B, parameter, radius, sdf)
    if clip_zero:
        data = np.where(data < 0, 0, data)
    return data

# --- top-level helper (must be outside the main function!) ---
def _compute_rounded_sdf_task(args):
    width, height, origin, destination, parameters_i, radius_i, clip_zero, sdf = args
    return compute_rounded_sdf(width, height, origin, destination,
                               parameters_i, radius_i, clip_zero, sdf)

def compute_rounded_sdf_multi(width: int,
                              height: int,
                              origin: tuple[int, int],
                              destination: tuple[int, int],
                              parameters: np.array,
                              radii: np.array,
                              clip_zero: bool,
                              sdf: Callable, 
                              count: int) -> np.ndarray:
    if count != len(parameters):
        raise ValueError(f'Expected {count} paramteres but got {len(parameters)}')
    if count != len(radii):
        raise ValueError(f'Expected {count} radii but got {len(radii)}')

    # prepare arguments for each process
    task_args = [
        (width, height, origin, destination, parameters[i], radii[i], clip_zero, sdf)
        for i in range(count)
    ]

    with ProcessPoolExecutor(max_workers=20) as executor:
        sdfs = list(executor.map(_compute_rounded_sdf_task, task_args))

    '''
    sdfs = []
    for i in tqdm(range(count)): 
        sdfs.append(compute_rounded_sdf(width, height, origin, destination, parameters[i], radii[i], clip_zero, sdf))
    '''
    
    return np.array(sdfs)