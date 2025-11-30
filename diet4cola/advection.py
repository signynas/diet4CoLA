import numpy as np

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

def interp_bilinear_cormack(phi: np.ndarray, xq: np.ndarray, yq: np.ndarray):
    """
    Bilinear interpolation of 2D array phi at query coords (xq, yq).
    xq, yq are floating point arrays with same shape (n_y, n_x).
    Coordinates are in pixel indices where x in [0, n_x-1], y in [0, n_y-1].

    Returns:
      vals: interpolated values
      neigh_vals: 4-neighbor values stacked with shape (..., 4) in order
                  [phi(y0,x0), phi(y0,x1), phi(y1,x0), phi(y1,x1)]
      (useful for monotonicity limiter)
    """
    n_y, n_x = phi.shape

    # Clip query coordinates to domain (we'll sample border values when query is outside)
    xq_clip = np.clip(xq, 0, n_x - 1)
    yq_clip = np.clip(yq, 0, n_y - 1)

    x0 = np.floor(xq_clip).astype(np.int64)
    y0 = np.floor(yq_clip).astype(np.int64)
    x1 = np.minimum(x0 + 1, n_x - 1)
    y1 = np.minimum(y0 + 1, n_y - 1)

    wx = xq_clip - x0
    wy = yq_clip - y0

    # Gather neighbor values
    Ia = phi[y0, x0]  # top-left
    Ib = phi[y0, x1]  # top-right
    Ic = phi[y1, x0]  # bottom-left
    Id = phi[y1, x1]  # bottom-right

    vals = (1 - wx) * (1 - wy) * Ia + wx * (1 - wy) * Ib + (1 - wx) * wy * Ic + wx * wy * Id

    neigh_vals = np.stack([Ia, Ib, Ic, Id], axis=-1)  # shape (..., 4)
    return vals, neigh_vals

def advect_backward(phi: np.ndarray,
                    v_x: np.ndarray,
                    v_y: np.ndarray,
                    magnitude: np.ndarray,
                    dt: float) -> np.ndarray:
    n_y, n_x = phi.shape
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

    # 1 - Backtrace positions
    x_prev = x - (dt * v_x * magnitude)
    y_prev = y - (dt * v_y * magnitude)

    # 2 - Interpolate old phi bilinearly at the backtraced position
    phi_new = interp_bilinear(phi, x_prev, y_prev)

    return phi_new

def advect_mac_cormack(phi: np.ndarray,
                       v_x: np.ndarray,
                       v_y: np.ndarray,
                       magnitude: np.ndarray,
                       dt: float,
                       apply_limiter: bool = True) -> np.ndarray:
    """
    Semi-Lagrangian MacCormack advection for scalar field phi.

    Inputs:
      phi: (n_y, n_x) array of scalar values at time t
      v_x, v_y: (n_y, n_x) velocity components (unit: pixels / time) or normalized
      magnitude: (n_y, n_x) additional per-pixel speed multiplier (kept for API compatibility)
      dt: timestep
      apply_limiter: if True, apply monotonicity limiter to corrected values

    Returns:
      phi_new: (n_y, n_x) advected field at time t + dt
    """
    n_y, n_x = phi.shape
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

    # 1) Predictor (semi-Lagrangian pullback: where did each target pixel come from?)
    x_prev = x - dt * v_x * magnitude
    y_prev = y - dt * v_y * magnitude

    phi_pred, neigh_pred = interp_bilinear_cormack(phi, x_prev, y_prev)

    # 2) Reverse/backward advection from predictor to estimate what we'd get forward-back
    #    We need to backtrace from the predicted field using -v to compute phi_back
    #    (i.e., where would the predictor's pixels have come from in phi_pred if we step -dt)
    x_prev_rev = x - dt * (-v_x) * magnitude  # equivalently x + dt * v_x * magnitude
    y_prev_rev = y - dt * (-v_y) * magnitude

    # interp phi_pred at x_prev_rev,y_prev_rev
    # but interp_bilinear currently samples from 'phi' array; we need to sample phi_pred,
    # so call interp_bilinear with phi_pred; but interp_bilinear expects the source grid,
    # so use phi_pred as the array and sample at x_prev_rev,y_prev_rev.
    # Note: phi_pred is a numpy array with same shape, so it works.
    phi_back, _ = interp_bilinear_cormack(phi_pred, x_prev_rev, y_prev_rev)

    # 3) Corrector (MacCormack): corrected = phi + 0.5*(phi_pred - phi_back)
    phi_corr = phi + 0.5 * (phi_pred - phi_back)

    if apply_limiter:
        # Monotonicity limiter: clamp phi_corr to min/max of source-stencil used in prediction.
        # We use neigh_pred (the 4 neighbors sampled when computing phi_pred).
        # min/max over the 4 neighbors
        neigh_min = np.min(neigh_pred, axis=-1)
        neigh_max = np.max(neigh_pred, axis=-1)

        # clamp
        phi_corr = np.minimum(np.maximum(phi_corr, neigh_min), neigh_max)

    # Return corrected field
    return phi_corr

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
    for i in range(iterations - 1):
        advected_i = advect_backward(advected[i], dx[i], dy[i], magnitude[i], step)
        advected.append(advected_i)
    return np.array(advected)

def advect_forward_sim(data: np.ndarray,
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
    for i in range(iterations - 1):
        advected_i = advect_mac_cormack(advected[i], dx[i], dy[i], magnitude[i], step)
        advected.append(advected_i)
    return np.array(advected)