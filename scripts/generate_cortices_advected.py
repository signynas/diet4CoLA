import argparse
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Imports from cola_advection.ipynb (Cell 2 & 4) ---
from diet4cola.advection import advect_backward_sim
from diet4cola.cut import CoLACut
from diet4cola.mask import mask_on_condition
# NOTE: directional_gradients is needed for SDF gradient calculation
from diet4cola.operations import invert, directional_gradients 
from diet4cola.velocity import compute_velocity_fields, prior_exponential
from diet4cola.sdf import sdf_box, compute_rounded_sdfs

# Assuming the Cortex and utility functions are available from the library
from diet4cola.cortex import Cortex, CortexSpec, generate_cortex_example
from diet4cola.utils import save_cortex, load_cortex

# --- Simulation Parameters (from notebook) ---
# Time steps
T_MIN = 0.0
T_MAX = 20.0  # Total simulation time
TIME_STEP = 0.5
TIMEPOINTS = np.arange(T_MIN, T_MAX + TIME_STEP, TIME_STEP)
ITERATIONS = len(TIMEPOINTS)
NORMALIZED_TIMEPOINTS = TIMEPOINTS / T_MAX

# Velocity/SDF Model parameters
K_PARAM = 0.25
VEL_PARAMETER = 20
SDF_RADIUS = 10
ALPHA = 5

# Grid parameters (from notebook Cell 3)
WIDTH = 512
HEIGHT = 512


def gen_advected_cortex_worker(idx: int, out: str, max_offset: int, extent_major: int,
                               extent_minor: int):
    """
    CPU-bound simulation worker: generates one cortex, simulates advection,
    and saves the time-series.
    """
    # 1. Generate Cortex (Initial State)
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
    
    # 2. Create CoLA Cut
    cola_cut = CoLACut((off_x, off_y), 32, 150, seed)

    # 3. Compute Time-Dependent SDF/Velocity Parameters
    initial_velocities = np.array([ALPHA * np.exp(-t * K_PARAM) for t in TIMEPOINTS])
    inverted_initial_velocities_norm = invert(initial_velocities / ALPHA)
    
    parameters = np.array([VEL_PARAMETER] * ITERATIONS)
    radii = [0] * ITERATIONS
    widths = [0] * ITERATIONS

    for i in range(ITERATIONS):
        if i == 0:
            radii[i] = 0
            widths[i] = 0
            continue
        radii[i] = inverted_initial_velocities_norm[i] * SDF_RADIUS
        widths[i] = widths[i - 1] + TIME_STEP * initial_velocities[i]

    # 4. Compute SDF and Velocity Fields (Aligned with Notebook Cells 106, 109, 113, 115)
    
    # 4a. Compute SDF fields (Notebook Cell 106)
    sdf_fields = compute_rounded_sdfs(WIDTH, HEIGHT, cola_cut.cut_origin, cola_cut.cut_destination, widths, radii, False, sdf_box, ITERATIONS)
    
    # 4b. Create SDF Masks and Clipped SDF (Notebook Cell 109)
    masked_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, 1) for sdf in sdf_fields]) 
    clipped_rounded_sdf_fields = np.array([mask_on_condition(sdf < 0, 0, sdf) for sdf in sdf_fields])

    # 4c. Compute raw velocity fields (Notebook Cell 113)
    velocity_fields = compute_velocity_fields(clipped_rounded_sdf_fields, prior_exponential, TIMEPOINTS, initial_velocities, parameters, ITERATIONS)
    
    # 4d. Create masked velocity fields and zero out first timestep (Notebook Cell 113, 115)
    masked_velocity_fields = np.array([vf * msdf for (vf, msdf) in zip(velocity_fields, masked_sdf_fields)])
    masked_velocity_fields[0, :, :] = np.where(clipped_rounded_sdf_fields[0, :, :] < 1, masked_velocity_fields[0, :, :], 0)
    
    # 4e. Compute SDF gradients (Notebook Cell 125) - **REQUIRED FOR ADVECTION**
    sdf_gradients = directional_gradients(clipped_rounded_sdf_fields)
    sdf_dys = sdf_gradients[:, 0, :, :]
    sdf_dxs = sdf_gradients[:, 1, :, :]

    # 5. Perform Backward Advection (Simulation) - **CORRECTED CALL**
    # Correct signature: advect_backward_sim(actomyosin_layer, sdf_dxs, sdf_dys, masked_velocity_fields, iterations, time_step)
    advected_cortex = advect_backward_sim(
        cortex.data, 
        sdf_dxs, 
        sdf_dys, 
        masked_velocity_fields, 
        ITERATIONS, 
        TIME_STEP
    )
    
    # 6. Save the full time-series (advected_cortex)
    np.save(Path(out) / f'cortex_{idx}_advected.npy', advected_cortex)

    # Return metadata including cut information
    return (idx, seed, cortex.cell_extent, cortex.cell_angle, spec.cell_blur_radius, cola_cut.cut_center, cola_cut.cut_origin, cola_cut.cut_destination, SDF_RADIUS, VEL_PARAMETER, T_MAX)


def main():
    parser = argparse.ArgumentParser(description="Parallel Cortex Generator with Advection")
    parser.add_argument("--n", type=int, required=True, help="Total number of cortices to generate")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for parallel processing")
    parser.add_argument("--out", type=str, default="./data", help="Output directory for generated data")
    parser.add_argument("--max_offset", type=int, default=100, help="Max offset from center for cell placement")
    parser.add_argument("--extent_long", type=int, default=450, help="Max extent for the major axis of the cell")
    parser.add_argument("--extent_short", type=int, default=300, help="Max extent for the minor axis of the cell")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)

    # Prepare tasks
    tasks = [(i, args.out, args.max_offset, args.extent_long, args.extent_short) for i in range(args.n)]

    results = [0] * args.n
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(gen_advected_cortex_worker, *t) for t in tasks]

        # Proper progress bar with as_completed
        for f in tqdm(as_completed(futures), total=args.n, desc="Generating advected cortices"):
            result = f.result()
            results[result[0]] = result

    # Define a custom header for the new metadata
    csv_header = ["cortex_id", "seed", "extent_major", "extent_minor", "angle", "radius", "cut_center_x", "cut_center_y", "cut_origin_x", "cut_origin_y", "cut_destination_x", "cut_destination_y", "sdf_radius", "vel_parameter", "t_max"]

    # Transform the results to match the header
    csv_rows = []
    for cortex in tqdm(results, desc='Saving metadata'):
        # The worker now returns more values: (idx, seed, extent, angle, blur_radius, cut_center, cut_origin, cut_destination, sdf_radius, vel_parameter, t_max)
        row = {
            "cortex_id": cortex[0],
            "seed": cortex[1],
            "extent_major": (cortex[2])[0],
            "extent_minor": (cortex[2])[1],
            "angle": cortex[3],
            "radius": cortex[4],
            "cut_center_x": cortex[5][0],
            "cut_center_y": cortex[5][1],
            "cut_origin_x": cortex[6][0],
            "cut_origin_y": cortex[6][1],
            "cut_destination_x": cortex[7][0],
            "cut_destination_y": cortex[7][1],
            "sdf_radius": cortex[8],
            "vel_parameter": cortex[9],
            "t_max": cortex[10],
        }
        csv_rows.append(row)

    # Write the CSV file
    csv_out_path = Path(args.out) / "cortices_metadata.csv"
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)

if __name__ == '__main__':
    # Define simulation parameters as constants in the script's global scope
    # (These are needed inside main and the worker for time array generation)
    main()