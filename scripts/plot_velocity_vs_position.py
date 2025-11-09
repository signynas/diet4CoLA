#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_velocity_vs_position(df, cell_ids, output_path=None, is_combined=False):
    """
    Plot velocity at time 0 vs position along x_parallel as histograms for specified cell_ids.
    Produces:
      - 2D heatmap (velocity vs. position)
      - Velocity histogram
      - Count distribution along x_parallel

    Args:
        df: DataFrame with columns [point_id, cell_id, time, velocity, x_parallel]
        cell_ids: List of cell_ids to plot
        output_path: Path prefix to save figures (if None, displays instead)
        is_combined: If True, combine multiple cells in one plot
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

    # Filter for first frame (time == 1 after reset)
    df_time1 = df_filtered[df_filtered['time'] == 1].copy()
    df_time1 = df_time1.dropna(subset=['velocity', 'x_parallel'])

    if df_time1.empty:
        print("No valid data at time 1 with non-negative x_parallel and velocity.")
        return

    # Determine binning for histogram
    x_range = df_time1['x_parallel'].max() - df_time1['x_parallel'].min()
    v_range = df_time1['velocity'].max() - df_time1['velocity'].min()
    n_bins_x = max(15, min(40, int(x_range / 3)))
    n_bins_v = max(10, min(30, int(v_range / 0.5)))

    # Bin data along x_parallel for averages
    bins = np.linspace(df_time1['x_parallel'].min(), df_time1['x_parallel'].max(), n_bins_x)
    df_time1['x_bin'] = pd.cut(df_time1['x_parallel'], bins=bins, include_lowest=True)

    # Compute mean velocity per x-bin
    bin_data = df_time1.groupby('x_bin', observed=True).agg({
        'x_parallel': 'mean',
        'velocity': 'mean'
    }).dropna().sort_values('x_parallel')

    # Count number of points per bin
    bin_counts = df_time1.groupby('x_bin', observed=True).size().reset_index(name='count')
    bin_counts['x_parallel'] = df_time1.groupby('x_bin', observed=True)['x_parallel'].mean().values
    bin_counts = bin_counts.sort_values('x_parallel')

    # Prepare output file paths
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        path_2d = os.path.join(output_dir, f"{base_name}_2d.png")
        path_vel = os.path.join(output_dir, f"{base_name}_velocity.png")
        path_count = os.path.join(output_dir, f"{base_name}_count.png")
    else:
        path_2d = path_vel = path_count = None

    # Title setup
    cell_str = ', '.join(map(str, cell_ids))
    title_prefix = 'Initial Velocity Analysis Across All Cells' if is_combined else f'Initial Velocity Analysis for Cell ID: {cell_str}'

    # === Plot 1: 2D Heatmap ===
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    h, xedges, yedges = np.histogram2d(df_time1['x_parallel'], df_time1['velocity'], bins=[n_bins_x, n_bins_v])
    im = ax1.imshow(h.T, origin='lower', aspect='auto', cmap='Blues',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Count', fontsize=11)

    # Overlay mean velocity
    ax1.plot(bin_data['x_parallel'], bin_data['velocity'],
             color='red', linewidth=2.5, label='Average', marker='o', markersize=5)

    ax1.set_xlabel('Position along x_parallel', fontsize=12)
    ax1.set_ylabel('Velocity at time 0 (pixels/time)', fontsize=12)
    ax1.set_title(f'{title_prefix}\n2D Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', color='white', linewidth=0.5)

    plt.tight_layout()
    if path_2d:
        plt.savefig(path_2d, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved 2D plot: {path_2d}")
    else:
        plt.show()

    # === Plot 2: Velocity Histogram ===
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(df_time1['velocity'], bins=n_bins_v,
             alpha=0.7, edgecolor='black', linewidth=0.5)

    mean_vel = df_time1['velocity'].mean()
    ax2.axvline(mean_vel, color='red', linewidth=2.5, linestyle='--', label=f'Mean: {mean_vel:.2f}')
    ax2.set_xlabel('Velocity at time 1 (pixels/time)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'{title_prefix}\nVelocity Distribution', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    if path_vel:
        plt.savefig(path_vel, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved velocity histogram: {path_vel}")
    else:
        plt.show()

    # === Plot 3: Count Distribution along x_parallel ===
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.bar(bin_counts['x_parallel'], bin_counts['count'],
            width=x_range / n_bins_x * 0.8,
            alpha=0.7, edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Position along x_parallel', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'{title_prefix}\nCount Distribution along x_parallel', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    if path_count:
        plt.savefig(path_count, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved count plot: {path_count}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot initial velocity vs position along x_parallel as histograms"
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

    # Determine which cells to plot
    if args.cell_ids:
        cell_ids = args.cell_ids
    else:
        cell_ids = df['cell_id'].unique().tolist()
        print(f"No cell_ids specified, plotting all {len(cell_ids)} cells")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Combined plot
    if not args.individual and len(cell_ids) > 1:
        print(f"\nCreating combined plot with all {len(cell_ids)} cells...")
        combined_output = os.path.join(args.output, "velocity_vs_position_all_cells.png") if args.output else None
        plot_velocity_vs_position(df, cell_ids, combined_output, is_combined=True)
    else:
        # Individual plots
        for idx, cell_id in enumerate(cell_ids, 1):
            print(f"\nPlotting cell {idx} of {len(cell_ids)}: {cell_id}")
            output_path = os.path.join(args.output, f"velocity_vs_position_{cell_id}.png") if args.output else None
            plot_velocity_vs_position(df, [cell_id], output_path)


if __name__ == "__main__":
    main()
