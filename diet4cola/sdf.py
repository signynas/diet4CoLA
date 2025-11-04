import numpy as np

from numpy import array, clip, dot, stack
from numpy.linalg import norm

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

def sdf_ellipse(P: np.ndarray,
                A: np.ndarray, 
                B: np.ndarray, 
                w: float) -> float:
    # Center of ellipse
    C = 0.5 * (A + B)
    # Major axis vector and length
    AB = B - A
    L = np.linalg.norm(AB) * 0.5  # half-length of major axis
    if L == 0:
        return np.linalg.norm(P - C) - w  # degenerate case: circle

    # Normalize major axis
    u = AB / (2 * L)
    # Construct rotation matrix to ellipse frame
    rot = np.array([[u[0], u[1]], [-u[1], u[0]]])

    # Rotate point into ellipse frame (centered)
    p_local = P - C
    p_rot = rot.T @ p_local  # do NOT use abs; keep smooth coordinates

    # Ellipse SDF (approximation using simple formula)
    x = p_rot[0] / L
    y = p_rot[1] / w
    length_xy = np.sqrt(x**2 + y**2)
    return length_xy - 1.0  # normalized distance; 0 at ellipse boundary