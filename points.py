#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np

def extract_point_xy_time(df, cell_id, point_id_offset):
    """
    Extracts point data from a DataFrame corresponding to a single .lineage file.
    Adds a unique point_id for each unique cell name in the file.

    Args:
    - df: DataFrame with columns ["cell", "time", "x", "y"]
    - cell_id: Identifier for the cell (derived from filename)
    - point_id_offset: Integer offset to ensure unique point_ids across multiple files
    Returns:
    - DataFrame with columns ["point_id", "cell_id", "cut", "time", "x", "y"]
    - Updated point_id_offset for next use
    """

    # Ensure required columns are present
    required = {"cell", "time", "x", "y"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    
    # Filter only valid cells (starting with "New nucleus" or "cut")
    valid_mask = df["cell"].str.startswith("New nucleus") | df["cell"].str.startswith("cut")
    df = df[valid_mask].copy()
    
    if df.empty:
        raise ValueError("No valid points found (\"cell\" field must start with 'New nucleus' or 'cut')")
    
    # Extract only needed columns
    result = df[["cell", "time", "x", "y"]].copy()
    
    # Determine cut as True/False based on cell name
    result["cut"] = result["cell"].apply(
        lambda x: x.startswith("cut")
    )
    
    # Add cell_id from filename
    result["cell_id"] = cell_id
    
    # Create unique point_id for each unique cell name within this file
    # Group by cell name to assign consistent point_ids
    cell_names = result["cell"].unique()
    cell_to_point_id = {name: point_id_offset + i for i, name in enumerate(cell_names)}
    result["point_id"] = result["cell"].map(cell_to_point_id)
    
    # Drop the original cell column
    result = result.drop(columns=["cell"])
    
    # Return result and the next available point_id
    return result, point_id_offset + len(cell_names)

def calculate_velocity_vectors(df):
    """
    Calculate velocity vectors (vx, vy, velocity) for each point based on previous position.
    velocity is calculated as displacement divided by time difference.

    Args:
    - df: DataFrame with columns ["point_id", "time", "x", "y"]
    Returns:
    - DataFrame with additional columns ["vx", "vy", "velocity"]
    """

    # Sort by point_id and time to ensure proper ordering
    df = df.sort_values(["point_id", "time"]).reset_index(drop=True)
    
    # Calculate differences within each point_id group
    df["dx"] = df.groupby("point_id")["x"].diff()
    df["dy"] = df.groupby("point_id")["y"].diff()
    df["dt"] = df.groupby("point_id")["time"].diff()
    
    # Calculate velocity components (avoid division by zero)
    df["vx"] = np.where(df["dt"] > 0, df["dx"] / df["dt"], np.nan)
    df["vy"] = np.where(df["dt"] > 0, df["dy"] / df["dt"], np.nan)
    
    # Calculate velocity (magnitude of velocity vector)
    df["velocity"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    
    # Drop temporary columns
    df = df.drop(columns=["dx", "dy", "dt"])
    
    return df

def parse_folder(folder):
    """
    Parse all .lineage files in the specified folder and compile into a single DataFrame.

    Args:
    - folder: Path to the folder containing .lineage files
    Returns:
    - DataFrame with columns ["point_id", "cell_id", "cut", "time", "x", "y", "vx", "vy", "velocity"]
    """

    pattern = os.path.join(folder, "*.lineage")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise SystemExit(f"No .lineage files found in {folder!r}")
    
    out_rows = []
    point_id_counter = 0
    
    for p in files:
        # Extract cell_id from filename (without .lineage extension)
        cell_id = os.path.splitext(os.path.basename(p))[0]
        
        try:
            df = pd.read_csv(p, sep="\t", comment="#", engine="python")
        except Exception as e:
            print(f"Warning: failed to read {p!r}: {e}", file=sys.stderr)
            continue
        
        try:
            sub, point_id_counter = extract_point_xy_time(df, cell_id, point_id_counter)
        except Exception as e:
            print(f"Warning: skipping {p!r}: {e}", file=sys.stderr)
            continue
        
        out_rows.append(sub)
    
    if not out_rows:
        raise SystemExit("No valid data parsed from .lineage files.")
    
    all_df = pd.concat(out_rows, ignore_index=True)
    
    # Calculate velocity vectors
    all_df = calculate_velocity_vectors(all_df)
    
    # Reorder columns: point_id, cell_id, cut, time, x, y, vx, vy, velocity
    all_df = all_df[["point_id", "cell_id", "cut", "time", "x", "y", "vx", "vy", "velocity"]]
    
    return all_df

def main():
    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Parse .lineage files and output CSV with point_id,cell_id,cut,time,x,y,vx,vy,velocity")
    p.add_argument("folder", help="Folder containing .lineage files")
    p.add_argument("-o", "--out", help="Output CSV file")
    args = p.parse_args()
    
    df = parse_folder(args.folder)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()