import numpy as np

from dataclasses import dataclass
from diet4cola.mask import cell_mask
from diet4cola.noise import actin_noise, myosin_noise
from diet4cola.operations import add, mul, normalize

@dataclass
class CortexSpec:
    seed: int                               = 42
    width: int                              = 512
    height: int                             = 512
    max_center_offset: int                  = 32
    max_cell_extent: tuple[int, int]        = (200, 200)
    cell_blur_radius: int                   = 50
    myosin_spot_scale: float                = 2.75
    myosin_spot_iterations: int             = 10
    myosin_noise_scale: float               = 0.02
    actin_resolution: int                   = 2560
    actin_myosin_offset: float              = -0.05

@dataclass
class Cortex:
    data: np.ndarray                        = None
    cell_angle: float                       = 0
    cell_extent: tuple[int, int]            = (0, 0)

def generate_cortex_example(spec: CortexSpec) -> Cortex:
    seed = spec.seed
    width = spec.width
    height = spec.height
    max_center_offset = spec.max_center_offset
    max_center_offset_half = max_center_offset // 2
    max_cell_extent = spec.max_cell_extent
    cell_blur_radius = spec.cell_blur_radius
    myosin_spot_scale = spec.myosin_spot_scale
    myosin_spot_iterations = spec.myosin_spot_iterations
    myosin_noise_scale = spec.myosin_noise_scale
    actin_resolution = spec.actin_resolution
    actin_myosin_offset = spec.actin_myosin_offset

    # 1. Generate cell center offset
    np.random.seed(seed)
    off_x = 256 + (np.random.randint(max_center_offset) - max_center_offset_half)
    off_y = 256 + (np.random.randint(max_center_offset) - max_center_offset_half)

    # 2. Generate cell mask (i.e. ellipse axes and angle)
    mask, extent, cell_angle = cell_mask(width, height, [off_x, off_y], max_cell_extent, cell_blur_radius, seed)

    # 3. Generate the myosin and actin layers
    myosin_layer = myosin_noise(width, height, myosin_spot_scale, myosin_noise_scale, myosin_spot_iterations, seed)
    actin_layer = actin_noise(width, height, mask, actin_resolution, actin_myosin_offset, seed)

    # 4. Multiply myosin/actin layers with the cell mask
    myosin_layer = mul(myosin_layer, mask)
    actin_layer = mul(actin_layer, mask)

    # 5. Add and normalize myosin and actin layers together
    actomyosin_layer = add(myosin_layer, actin_layer)
    actomyosin_layer = normalize(actomyosin_layer)

    # Return all data
    return Cortex(actomyosin_layer, cell_angle, extent)