import numpy as np

from diet4cola.operations import *
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

def get_snoise_generator(seed: int = 42):
    return OpenSimplex(seed)

def snoise(generator: OpenSimplex, 
           x: int, 
           y: int, 
           scale: float = 1.0) -> float:
    n_x = x * scale
    n_y = y * scale
    return generator.noise2(n_x, n_y)

def fractal_snoise(generator: OpenSimplex, 
                   width: int,
                   height: int,
                   scale: float = 1.0,
                   octaves: int = 8,
                   gain: float = 0.5,
                   lacunarity: float = 2.0,
                   normalized: bool = True) -> np.ndarray:
    data = np.zeros((height, width))
    yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    for i in range(octaves):
        frequency = scale * (lacunarity ** i)
        amplitude = gain ** i

        # Compute coordinates scaled by frequency
        xs = xv * frequency
        ys = yv * frequency
        
        # Flatten, compute noise, then reshape
        noise = np.array([generator.noise2(x, y) for x, y in zip(xs.ravel(), ys.ravel())])
        data += amplitude * noise.reshape(height, width)

    if normalized:
        data = (data - data.min()) / (data.max() - data.min())
    return data

def ridge_snoise(generator: OpenSimplex, 
                 width: int,
                 height: int,
                 scale: float = 1.0,
                 octaves: int = 8,
                 gain: float = 0.5,
                 lacunarity: float = 2.0,
                 normalized: bool = True) -> np.ndarray:
    data = np.zeros([height, width])
    frequency = scale
    amplitude = 1.0

    for i in range(octaves):
        for y in range(height):
            for x in range(width):
                n_x = x * frequency
                n_y = y * frequency
                data[y, x] += amplitude * abs(generator.noise2(n_x, n_y))
        
        amplitude *= gain
        frequency *= lacunarity

    if normalized:
        data = (data - data.min()) / (data.max() - data.min())
    return data

def worley_noise(width: int,
                 height:int, 
                 num_points: int,
                 seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * [width, height]

    # Query grid coordinates in batches
    y, x = np.mgrid[0:height, 0:width]
    coords = np.dstack((x, y))

    # Query nearest distance for each pixel (slow)
    dist = np.min(np.sqrt(((coords[:, :, None, :] - points) ** 2).sum(axis=3)), axis=2)
    return dist / dist.max()

def worley_noise_kd(width: int,
                      height: int,
                      num_points: int,
                      seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * [width, height]

    # Build KDTree (O(N log N))
    tree = cKDTree(points)

    # Query grid coordinates in batches
    y, x = np.mgrid[0:height, 0:width]
    coords = np.column_stack((x.ravel(), y.ravel()))

    # Query nearest distance for each pixel (fast!)
    dist, _ = tree.query(coords, k=1)
    dist = dist.reshape((height, width))
    return dist / dist.max()

def spot_noise(width: int,
               height: int,
               num_points: int,
               sigma: float = 1.0,
               amplitude: float = 1.0,
               seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0: width]
    x = x / width
    y = y / width

    # Initialize empty texture
    data = np.zeros((height, width), dtype=np.float32)

    # Sample random spot centers
    cx = rng.random(num_points)
    cy = rng.random(num_points)

    # Random amplitudes for variation
    amps = amplitude * (0.5 + rng.random(num_points))

    for i in range(num_points):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx * dx + dy * dy
        blob = amps[i] * np.exp(-dist_sq / (2 * sigma ** 2))
        data += blob

    return normalize(data)

def spot_noise_convolved(width: int,
                         height: int,
                         num_points: int,
                         sigma: float = 1.0,
                         amplitude: float = 1.0,
                         seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Initialize empty impulse map
    data = np.zeros((height, width), dtype=np.float32)

    # Random spot positions
    xs = rng.integers(0, width, size=num_points)
    ys = rng.integers(0, height, size=num_points)

    # Random amplitudes for variation
    amps = amplitude * (0.5 + rng.random(num_points))

    # Place impulses
    for x, y, a in zip(xs, ys, amps):
        data[y, x] += a

    # Convolve impulses with Gaussian to get smooth blobs
    blurred = gaussian_filter(data, sigma=sigma, mode='wrap')

    # Normalize
    blurred -= blurred.min()
    blurred /= blurred.max() + 1e-8

    return blurred


def myosin_noise(width: int,
                 height: int, 
                 spot_scale: float = 2.75,
                 scale: float = 0.02, 
                 iterations: int = 10,
                 seed: int = 42) -> np.ndarray:
    data = np.zeros([width, height])
    for i in range(iterations):
        seed_i = seed + i
        mask_i_0    = spot_noise_convolved(width, height, num_points=100, sigma=spot_scale, amplitude=1, seed=seed_i)
        mask_i_90   = rotate(mask_i_0, 90)
        mask_i_180  = rotate(mask_i_0, 180)
        mask_i_270  = rotate(mask_i_0, 270)

        mask_i      = add(mask_i_0, add(mask_i_90, add(mask_i_180, mask_i_270)))
        data        = add(data, mask_i)
        data        = normalize(data)

    noise_mask      = fractal_snoise(get_snoise_generator(seed), width, height, scale=scale)
    data            = mul(data, noise_mask)

    return normalize(data)

def actin_noise(width: int,
                height: int,
                cell_mask: np.ndarray,
                num_points: int,
                myosin_offset: float = -0.05,
                seed: int = 42) -> np.ndarray:
    # Worley mask
    worley_mask     = worley_noise_kd(width, height, num_points, seed)
    worley_mask     = power(worley_mask, 2)
    worley_mask     = sigmoid(worley_mask, 5, 0.25)
    worley_mask     = normalize(worley_mask)

    # Worley Gradient
    dy, dx          = np.gradient(worley_mask)
    dx              = normalize(dx)
    dy              = normalize(dy)
    dxdy            = normalize(mul(dx, dy))
    dxdy_inv        = invert(dxdy)

    # Edge mask
    edge_mask       = canny(dxdy, 0, 255)
    edge_mask       = blur(edge_mask, (5, 5), 3)
    edge_mask       = normalize(edge_mask)

    edge_mask_inv   = canny(dxdy_inv, 0, 255)
    edge_mask_inv   = blur(edge_mask_inv, (5, 5), 3)
    edge_mask_inv   = normalize(edge_mask_inv)

    final_edge_mask = normalize(add(edge_mask, edge_mask_inv))
    final_edge_mask = blur(final_edge_mask, (11, 11), 3)
    final_edge_mask = normalize(final_edge_mask)

    # Myosin mask
    myosin_mask      = myosin_noise(width, height, 0.0035, 0.02, 5, seed + 1)
    myosin_mask      = normalize(blur(myosin_mask, (3, 3), 2))
    myosin_layer     = sigmoid_offset(cell_mask, 10, 0.5, myosin_offset)
    myosin_mask      = normalize(mul(myosin_mask, myosin_layer))

    # Simplex mask
    simplex_layer   = fractal_snoise(get_snoise_generator(seed + 3), width, height, 0.01)
    simplex_mask    = mul(final_edge_mask, simplex_layer)

    # Simplex detail
    simplex_detail  = fractal_snoise(get_snoise_generator(seed + 4), width, height, 0.05, 4)

    # Sigmoid mask
    sigmoid_mask    = sigmoid(cell_mask, 10, 0.8)

    # Layer composition
    composition     = mul(final_edge_mask, simplex_mask)
    composition     = normalize(add(composition, simplex_mask))
    composition     = lerp(composition, simplex_detail, 0.85)
    composition     = normalize(add(composition, myosin_mask))
    composition     = mul(composition, sigmoid_mask)
    composition     = normalize(composition)
    composition     = sigmoid(composition, 2)

    return normalize(composition)