import numpy as np

from diet4cola.generation.operations import blur

def ellipse(width: int,
            height: int,
            center: tuple[int, int],
            axes: tuple[int, int],
            angle_deg: float = 0.0) -> np.ndarray:
    cy, cx = center
    a, b = axes
    angle_rad = np.deg2rad(angle_deg)

    # Create coordinate grid
    y, x = np.ogrid[:width, :height]

    # Translate grid to center
    x_shift = x - cx
    y_shift = y - cy

    # Rotate coordinates by -angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    x_rot = x_shift * cos_angle + y_shift * sin_angle
    y_rot = -x_shift * sin_angle + y_shift * cos_angle

    # Equation of ellipse: (x/a)^2 + (y/b)^2 <= 1
    mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
    return mask * 1.0

def cell_mask(width: int,
              height: int,
              offset: tuple[int, int], 
              extent: tuple[int, int] = (200, 200),
              sigma: float = 1.0,
              seed: int = 42) -> tuple[np.ndarray, tuple[int, int], float]:
    max_a, max_b = extent

    np.random.seed(seed + 3)
    a = np.random.rand() * max_a
    b = np.random.rand() * max_b
    angle = np.random.rand() * 360

    kernel = (width // 4 - 1, height // 4 - 1)
    cell_mask = ellipse(width, height, offset, (a, b), angle)    
    cell_mask = blur(cell_mask, kernel, sigma)

    return cell_mask, (a, b), angle

def cut_mask(width: int,
             height: int, 
             offset: tuple[int, int],
             extent: tuple[int, int],
             angle: float) -> np.ndarray:
    return ellipse(width, height, offset, extent, angle)

def mask_on_condition(condition: np.ndarray,
                      true_value: float,
                      false_value: float) -> np.ndarray:
    return np.where(condition, true_value, false_value)