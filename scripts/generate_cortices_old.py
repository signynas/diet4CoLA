import argparse
import csv
import numpy as np

from diet4cola.cortex import Cortex, CortexSpec, generate_cortex_example
from diet4cola.utils import save_cortex

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, Lock
from pathlib import Path
from threading import Thread
from tqdm import tqdm

# Shared progress lock for tqdm
from threading import Lock as ThreadLock

progress_lock = ThreadLock()
STOP_SIGNAL = "STOP"

def writer_worker(queue: Queue, out: Path, csv_path: Path, stop_signal, lock, pbar: tqdm):
    """
    Writer thread: consumes (Cortex, idx) from the queue,
    writes the .npy file and appends metadata.
    """
    metadata_folder = out / "metadata"
    metadata_folder.mkdir(parents=True, exist_ok=True)

    # Initialize CSV if it does not exist
    with lock:
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "extent_major", "extent_minor", "cell_angle"])

    while True:
        item = queue.get()

        if item == STOP_SIGNAL:
            break

        cortex_i, idx = item
        filename = f"cortex_{idx}.npy"
        save_cortex(cortex_i.data, out / filename)

        with lock:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename,
                                 int(cortex_i.cell_extent[0]),
                                 int(cortex_i.cell_extent[1]),
                                 float(cortex_i.cell_angle)])
        with progress_lock:
            pbar.update(1)

def gen_cortex_worker(idx: int, out: Path, max_offset: int, extent_major: int,
                      extent_minor: int, queue: Queue):
    """
    CPU-bound cortex generator: creates one Cortex and submits to the queue.
    """
    spec = CortexSpec()
    spec.seed = np.random.randint(0, 1e8)
    spec.max_center_offset = max_offset
    spec.max_cell_extent = (extent_major, extent_minor)
    spec.cell_blur_radius = np.random.randint(45, 55)

    cortex = generate_cortex_example(spec)
    queue.put((cortex, idx))  # send to writer

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
    csv_path = out / "metadata" / "metadata.csv"

    queue = Queue(maxsize=args.workers * 2)  # multiprocessing-safe queue
    lock = Lock()  # multiprocessing-safe lock

    # Start the writer thread
    pbar = tqdm(total=args.n, desc="Cortices completed")
    writer_thread = Thread(target=writer_worker, args=(queue, out, csv_path, STOP_SIGNAL, lock, pbar))
    writer_thread.start()

    # Launch CPU-bound cortex generation in parallel processes
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx in range(args.n):
            executor.submit(gen_cortex_worker, idx, out, args.max_offset,
                            args.extent_long, args.extent_short, queue)

    # Signal writer to stop after all tasks are submitted
    queue.put(STOP_SIGNAL)

    # Wait for the writer thread to finish
    writer_thread.join()
    pbar.close()
    print(f"All {args.n} cortices generated and metadata saved to {csv_path}")

if __name__ == "__main__":
    main()
