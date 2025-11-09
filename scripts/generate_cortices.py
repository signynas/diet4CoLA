import argparse
import csv
import numpy as np

from diet4cola.cortex import Cortex, CortexSpec, generate_cortex_example
from diet4cola.utils import save_cortex

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def gen_cortex_worker(idx: int, out: str, max_offset: int, extent_major: int,
                      extent_minor: int):
    """
    CPU-bound cortex generator: creates one Cortex and submits to the queue.
    """
    spec = CortexSpec()
    spec.seed = np.random.randint(0, 1e8)
    spec.max_center_offset = max_offset
    spec.max_cell_extent = (extent_major, extent_minor)
    spec.cell_blur_radius = np.random.randint(45, 55)

    # Generate
    cortex = generate_cortex_example(spec)

    # Safe
    save_cortex(cortex.data, Path(out) / f'cortex_{idx}')

    return (idx, spec.seed, cortex.cell_extent, cortex.cell_angle, spec.cell_blur_radius)

def main():
    parser = argparse.ArgumentParser(description="Parallel Cortex Generator")
    parser.add_argument("--n", type=int, required=True, help="Total number of cortices to generate")
    parser.add_argument("--workers", type=int, required=True, help="Number of parallel CPU workers")
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    parser.add_argument("--max-offset", type=int, default=32)
    parser.add_argument("--extent-long", type=int, default=200)
    parser.add_argument("--extent-short", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)

    # Prepare tasks
    tasks = [(i, args.out, args.max_offset, args.extent_long, args.extent_short) for i in range(args.n)]

    results = [0] * args.n
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(gen_cortex_worker, *t) for t in tasks]

        # Proper progress bar with as_completed
        for f in tqdm(as_completed(futures), total=args.n, desc="Generating cortices"):
            result = f.result()
            results[result[0]] = result

    # Define a custom header (the order of columns you want)
    csv_header = ["cortex_id", "seed", "extent_major", "extent_minor", "angle", "radius"]

    # Transform the results to match the header
    # Example: extracting fields from your results dict and adding an ID
    csv_rows = []
    for cortex in tqdm(results, desc='Saving metadata'):
        row = {
            "cortex_id": cortex[0],
            "seed": cortex[1],
            "extent_major": (cortex[2])[0],
            "extent_minor": (cortex[2])[1],
            "angle": cortex[3],
            "radius": cortex[4]
        }
        csv_rows.append(row)

    # Write CSV
    csv_path = out / "metadata" / "metadata.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)

if __name__ == "__main__":
    main()
