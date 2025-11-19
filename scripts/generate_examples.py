import argparse
import csv
import numpy as np
import shortuuid

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from diet4cola.advection import advect_backward_sim
from diet4cola.cut import CoLACut
from diet4cola.cortex import Cortex, CortexSpec, generate_cortex_example
from diet4cola.mask import mask_on_condition
from diet4cola.operations import invert, directional_gradients 
from diet4cola.velocity import compute_velocity_fields, prior_exp
from diet4cola.sdf import sdf_box, compute_rounded_sdfs
from diet4cola.utils import save_array, load_array

# Grid parameters
WIDTH = 512
HEIGHT = 512

def initialize_velocity_exponential(alpha: float, t: float, k: float) -> float:
    return alpha * np.exp(-t * k)

def initialize_velocity_hyperbolic(alpha: float, t: float, k: float) -> float:
    return (alpha ** 2) * (1.0 / (k * t + alpha))

def gen_advected_cortex_worker(idx: int, out: str, step: float, max_offset: int, extent_major: int,
                               extent_minor: int) -> dict:
    # 1. Simulation Parameters
    t_min = 0.0
    t_max = 10.0  # Total simulation time (in seconds)
    time_step = step
    timepoints = np.arange(t_min, t_max + time_step, time_step)
    iterations = len(timepoints)
    normalized_timepoints = timepoints / t_max

    # 2. Generate Cortex (Initial State)
    spec = CortexSpec()
    seed = np.random.randint(0, 1e8)
    np.random.seed(seed)
    spec.seed = seed
    spec.max_center_offset = max_offset
    spec.max_cell_extent = (extent_major, extent_minor)
    spec.cell_blur_radius = np.random.randint(45, 55)
    cortex = generate_cortex_example(spec)

    # Calculate cut center (randomized offset as in notebook Cell 3)
    off_x = WIDTH // 2 + (np.random.randint(32) - 16)
    off_y = HEIGHT // 2 + (np.random.randint(32) - 16)
    
    # 3. Create CoLA Cut
    cola_cut = CoLACut((off_x, off_y), 32, 150, seed)

    # 4. Compute Time-Dependent SDF/Velocity Parameters
    # 4a. Sample initial parameters
    sdf_radius = np.random.rand() * 5 # Max SDF radius parameter of 5
    alpha = np.random.rand() * 10 # Max initial velocity of 10
    falloff_distance = np.random.randint(0, 150) # Max falloff distance of 150
    use_exponential = np.random.rand() > 0.5

    if use_exponential:
        k = np.random.rand() * 0.5
    else:
        k = np.random.rand() * 10

    # 4b. Create velocity profiles for the edge of the SDF
    initial_velocities = np.array([initialize_velocity_exponential(alpha, t, k) if use_exponential else initialize_velocity_hyperbolic(alpha, t, k) for t in timepoints])
    inverted_initial_velocities_normalized = invert(initial_velocities / alpha)
    
    # 4c. Create radius and width of the cut over time
    radii = [0] * iterations
    widths = [0] * iterations

    for i in range(iterations):
        if i == 0:
            radii[i] = 0
            widths[i] = 0
            continue

        radii[i] = inverted_initial_velocities_normalized[i] * sdf_radius
        widths[i] = widths[i - 1] + time_step * initial_velocities[i]

    # 5. Compute SDF and Velocity Fields
    # 5a. Compute SDF fields
    sdf_fields = compute_rounded_sdfs(WIDTH, HEIGHT, cola_cut.cut_origin, cola_cut.cut_destination, widths, radii, False, sdf_box, iterations)
    
    # 5b. Create SDF Masks and Clipped SDF
    masked_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, 1) for sdf in sdf_fields]) 
    clipped_rounded_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, sdf) for sdf in sdf_fields])

    # 5c. Compute raw velocity fields
    velocity_fields = compute_velocity_fields(clipped_rounded_sdf_fields, prior_exp, timepoints, initial_velocities, falloff_distance, iterations)
    
    # 5d. Create masked velocity fields and zero out first timestep
    masked_velocity_fields = np.array([vf * msdf for (vf, msdf) in zip(velocity_fields, masked_sdf_fields)])
    masked_velocity_fields[0, :, :] = np.where(clipped_rounded_sdf_fields[0, :, :] < 1, masked_velocity_fields[0, :, :], 0)
    
    # 5e. Compute SDF gradients
    sdf_gradients = directional_gradients(clipped_rounded_sdf_fields)
    sdf_dys = sdf_gradients[:, 0, :, :]
    sdf_dxs = sdf_gradients[:, 1, :, :]

    # 6. Perform Backward Advection 
    # 6a. Backward advection for the actomyosin cortex
    advected_cortex = advect_backward_sim(
        cortex.data, 
        sdf_dxs, 
        sdf_dys, 
        masked_velocity_fields, 
        iterations, 
        time_step
    )

    # 6b. Backward advection for the myosin and actin layers separately
    advected_myosin = advect_backward_sim(
        cortex.myosin_channel,
        sdf_dxs,
        sdf_dys,
        masked_velocity_fields,
        iterations, 
        time_step
    )

    advected_actin = advect_backward_sim(
        cortex.actin_channel,
        sdf_dxs,
        sdf_dys,
        masked_velocity_fields,
        iterations, 
        time_step
    )
    
    # 7. Save the advected fields and the velocity field
    # 7a. Generate a UUID for this cell, just have a unique identifier
    cell_uuid = shortuuid.uuid()

    # 7b. Save the advected cortex and its separate layers
    save_array(advected_cortex, Path(out) / f'cortex_{cell_uuid}_actomyosin.npy')
    save_array(advected_myosin, Path(out) / f'cortex_{cell_uuid}_myosin.npy')
    save_array(advected_actin, Path(out) / f'cortex_{cell_uuid}_actin.npy')

    # 7c. Save the velocity field that was used to advect the cortex
    save_array(masked_velocity_fields, Path(out) / f'cortex_{cell_uuid}_velocity_field.npy')

    # Return useful metadata
    return {
        # General metadata for reproduction
        'idx': idx,
        'seed': seed,

        # Cell metadata
        'cell_extent': cortex.cell_extent,
        'cell_angle': cortex.cell_angle,
        
        # Cut metadata
        'cut_off_x': off_x,
        'cut_off_y': off_y,
        'cut_off_bound': 32,
        'cut_length_limit': 150,

        # Velocity metadata
        'sdf_radius': sdf_radius,
        'initial_velocity': alpha,
        'falloff_distance': falloff_distance,
        'prior': 'exponential' if use_exponential else 'hyperbolic',

        # Simulation metadata
        'width': WIDTH,
        'height': HEIGHT,
        't_min': t_min,
        't_max': t_max,
        'step': time_step
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Cortex Generator with Advection")
    parser.add_argument("--n", type=int, required=True, help="Total number of cortices to generate")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for parallel processing")
    parser.add_argument("--out", type=str, default="./data", help="Output directory for generated data")
    parser.add_argument("--step", type=float, default=0.5, help="Timestep to use in the advection simulation")
    parser.add_argument("--max_offset", type=int, default=100, help="Max offset from center for cell placement")
    parser.add_argument("--extent_long", type=int, default=450, help="Max extent for the major axis of the cell")
    parser.add_argument("--extent_short", type=int, default=300, help="Max extent for the minor axis of the cell")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)

    # Prepare tasks
    tasks = [(i, args.out, args.step, args.max_offset, args.extent_long, args.extent_short) for i in range(args.n)]

    results = [0] * args.n
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(gen_advected_cortex_worker, *t) for t in tasks]

        # Proper progress bar with as_completed
        for f in tqdm(as_completed(futures), total=args.n, desc="Generating advected cortices"):
            result = f.result()
            results[result['idx']] = result

    # Define a custom header for the new metadata
    csv_header = [
        'idx', 
        'seed', 
        'cell_extent', 
        'cell_angle', 
        'cut_off_x',
        'cut_off_y',
        'cut_off_bound',
        'cut_length_limit',
        'sdf_radius',
        'initial_velocity',
        'falloff_distance',
        'prior',
        'width',
        'height',
        't_min',
        't_max',
        'step'
    ]

    # Transform the results to match the header
    csv_rows = []
    for cortex_metadata in tqdm(results, desc='Saving metadata'):
        csv_rows.append(cortex_metadata)

    csv_out_path = Path(args.out) / "cortices_metadata.csv"
    file_exists = csv_out_path.exists()

    with open(csv_out_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    print(f'Done.')


if __name__ == '__main__':
    main()