import numpy as np

from tqdm import tqdm

def interp_bilinear(data: np.ndarray,
                    x: float, 
                    y: float) -> float:
    # Compute possible grid coordinates
    x_0 = np.floor(x).astype(int)
    x_1 = x_0 + 1
    y_0 = np.floor(y).astype(int)
    y_1 = y_0 + 1

    # Clamp values to the grid (to prevent out of bounds!)
    height, width = data.shape  
    x_0 = np.clip(x_0, 0, width - 1)
    x_1 = np.clip(x_1, 0, width - 1)
    y_0 = np.clip(y_0, 0, height - 1)
    y_1 = np.clip(y_1, 0, height - 1)

    # Interpolation weights
    w_x = x - x_0
    w_y = y - y_0

    # Linearly interpolate first along x then along y
    top = (1 - w_x) * data[y_0, x_0] + w_x * data[y_0, x_1]
    bottom = (1 - w_x) * data[y_1, x_0] + w_x * data[y_1, x_1]
    return (1 - w_y) * top + w_y * bottom

def advect_backward(phi: np.ndarray,
                    v_x: np.ndarray,
                    v_y: np.ndarray,
                    magnitude: np.ndarray,
                    dt: float) -> float:
    n_y, n_x = phi.shape
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

    # 1 - Backtrace positions
    x_prev = x - (dt * v_x * magnitude)
    y_prev = y - (dt * v_y * magnitude)

    # 2 - Interpolate old phi bilinearly at the backtraced position
    phi_new = interp_bilinear(phi, x_prev, y_prev)

    return phi_new

def advect_backward_sim(data: np.ndarray,
                        dx: np.ndarray,
                        dy: np.ndarray, 
                        magnitude: np.ndarray,
                        iterations: int,
                        step: float) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a 2D NumPy array")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")
    if data.size == 0:
        raise ValueError("data cannot be empty")
    if dx.shape != dy.shape or dx.shape != magnitude.shape:
        raise ValueError(f"dX and dY must have the same shape")
    if dx.ndim != 3 or dy.ndim != 3 or magnitude.ndim != 3:
        raise ValueError(f'dX, dY and magnitude must be 3D, got {dx.ndim}D, {dy.ndim}D and {magnitude.ndim}D')

    advected = [data]
    for i in tqdm(range(iterations - 1)):
        advected_i = advect_backward(advected[i], dx[i], dy[i], magnitude[i], step)
        advected.append(advected_i)
    return np.array(advected)