#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_velocities(df, cell_ids, output_path=None, is_combined=False):
    """
    Plot velocity over time for specified cell_ids.
    Individual point trajectories shown in light grey, average in bold color.
    Time is reset to start from 0 for each cell.
    
    Args:
        df: DataFrame with columns [point_id, cell_id, time, velocity, vx, vy]
        cell_ids: List of cell_ids to plot
        output_path: Path to save figure (if None, displays instead)
        is_combined: If True, combine all cell_ids into one plot
    """
    # Filter data for specified cell_ids
    df_filtered = df[df['cell_id'].isin(cell_ids)].copy()

    if df_filtered.empty:
        print(f"No data found for cell_ids: {cell_ids}")
        return

    # Reset time to start from 0 for each cell_id
    for cell_id in cell_ids:
        cell_mask = df_filtered['cell_id'] == cell_id
        min_time = df_filtered.loc[cell_mask, 'time'].min()
        df_filtered.loc[cell_mask, 'time'] = df_filtered.loc[cell_mask, 'time'] - min_time

    # Drop NaN velocities (e.g. first frame)
    df_filtered = df_filtered.dropna(subset=['velocity'])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual trajectories
    if not is_combined:
        for point_id in df_filtered['point_id'].unique():
            point_data = df_filtered[df_filtered['point_id'] == point_id]
            ax.plot(point_data['time'], point_data['velocity'],
                    color='lightgrey', alpha=0.6, linewidth=1, zorder=1)

    # Compute average velocity across all points at each time
    avg_data = df_filtered.groupby('time')['velocity'].mean().reset_index()
    ax.plot(avg_data['time'], avg_data['velocity'], linewidth=2.5, label='Average', zorder=2)

    # Labels and styling
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Velocity (pixels/time)', fontsize=12)
    ax.set_title('Cell Velocity Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)

    # Format x-axis
    min_time = df_filtered['time'].min()
    max_time = df_filtered['time'].max()
    if max_time > min_time:
        ax.set_xticks(np.arange(int(np.floor(min_time)), int(np.ceil(max_time)) + 1, 1))
        ax.set_xlim(left=int(np.floor(min_time)))

    # Figure title
    cell_ids_str = ', '.join(map(str, cell_ids))
    if is_combined:
        fig.suptitle('Velocity Analysis Across All Cells', fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle(f'Velocity Analysis for Cell ID: {cell_ids_str}',
                     fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cell velocities with individual trajectories and averages"
    )
    parser.add_argument("input_csv", help="Input CSV file (filtered or raw)")
    parser.add_argument("-c", "--cell-ids", nargs='+',
                        help="Cell IDs to plot (space-separated)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory for plots")
    parser.add_argument("--individual", action="store_true",
                        help="Plot each cell_id separately instead of combined")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    # Determine which cell_ids to plot
    if args.cell_ids:
        cell_ids = args.cell_ids
    else:
        cell_ids = df['cell_id'].unique().tolist()
        print(f"No cell_ids specified, plotting all {len(cell_ids)} cells")

    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Combined plot
    if not args.individual and len(cell_ids) > 1:
        print(f"\nCreating combined plot for all {len(cell_ids)} cells...")
        combined_output = os.path.join(args.output, "velocity_all_cells.png") if args.output else None
        plot_velocities(df, cell_ids, combined_output, is_combined=True)
    else:
        # Individual plots
        for idx, cell_id in enumerate(cell_ids, 1):
            print(f"\nPlotting cell {idx} of {len(cell_ids)}: {cell_id}")
            output_path = os.path.join(args.output, f"velocity_{cell_id}.png") if args.output else None
            plot_velocities(df, [cell_id], output_path)


if __name__ == "__main__":
    main()
