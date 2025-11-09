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

def calculate_cut_coordinates(df):
    """
    Calculate transformed coordinates (x_parallel, x_perpendicular) based on cut axis.
    The cut axis is defined by two cut points within each cell_id.
    
    For each cell_id:
    - Find the two cut points
    - Define cut axis vector and perpendicular vector
    - Transform all points (cut and non-cut) to the new coordinate system
    - x_parallel: distance along the cut axis (normalized by cut length)
    - x_perpendicular: distance perpendicular to the cut axis
    
    Args:
    - df: DataFrame with columns ["cell_id", "cut", "time", "x", "y"]
    Returns:
    - DataFrame with additional columns ["x_parallel", "x_perpendicular", "cut_length"]
    """
    
    # Initialize new columns
    df["x_parallel"] = np.nan
    df["x_perpendicular"] = np.nan
    df["cut_length"] = np.nan
    
    # Process each cell_id separately
    for cell_id in df["cell_id"].unique():
        cell_mask = df["cell_id"] == cell_id
        cell_df = df[cell_mask]
        
        # Find cut points
        cut_points = cell_df[cell_df["cut"] == True]
            
        if len(cut_points) > 2:
            print(f"Warning: cell_id {cell_id!r} has more than 2 cut points; skipping cut coordinate calculation.", file=sys.stderr)
            continue

        if len(cut_points) < 2:
            # Need at least 2 cut points to define an axis
            print(f"Warning: cell_id {cell_id!r} has less than 2 cut points; skipping cut coordinate calculation.", file=sys.stderr)
            continue
        
        # Use the first two cut points to define the axis
        cut1 = cut_points.iloc[0]
        cut2 = cut_points.iloc[1]
        
        # Calculate cut axis vector
        cut_vector = np.array([cut2["x"] - cut1["x"], cut2["y"] - cut1["y"]])
        cut_length = np.linalg.norm(cut_vector)
        
        if cut_length == 0:
            # Degenerate case: both cut points are at the same location
            print(f"Warning: cell_id {cell_id!r} has identical cut points; skipping cut coordinate calculation.", file=sys.stderr)
            continue
        
        # Normalize cut vector
        cut_unit = cut_vector / cut_length
        
        # Perpendicular vector (rotate 90 degrees counter-clockwise)
        perp_unit = np.array([-cut_unit[1], cut_unit[0]])
        
        # Origin for transformation (use first cut point)
        origin = np.array([cut1["x"], cut1["y"]])
        
        # Transform all points in this cell
        for idx in cell_df.index:
            point = np.array([df.loc[idx, "x"], df.loc[idx, "y"]])
            relative_pos = point - origin
            
            # Project onto cut axis and perpendicular axis
            df.loc[idx, "x_parallel"] = np.dot(relative_pos, cut_unit) / cut_length
            df.loc[idx, "x_perpendicular"] = np.dot(relative_pos, perp_unit)
            df.loc[idx, "cut_length"] = cut_length
    
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
    
    # Calculate cut-based coordinate transformation
    all_df = calculate_cut_coordinates(all_df)
    
    # Reorder columns: point_id, cell_id, cut, time, x, y, vx, vy, velocity, x_parallel, x_perpendicular, cut_length
    all_df = all_df[["point_id", "cell_id", "cut", "time", "x", "y", "vx", "vy", "velocity", "x_parallel", "x_perpendicular", "cut_length"]]
    
    return all_df

def main():
    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Parse .lineage files and output CSV with point_id,cell_id,cut,time,x,y,vx,vy,velocity,x_parallel,x_perpendicular,cut_length")
    p.add_argument("folder", help="Folder containing .lineage files")
    p.add_argument("-o", "--out", help="Output CSV file")
    args = p.parse_args()
    
    df = parse_folder(args.folder)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()