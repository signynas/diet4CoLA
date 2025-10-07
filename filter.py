#!/usr/bin/env python3
import argparse
import pandas as pd

def filter_velocity_outliers(df, cell_ids, filter_threshold):
    """
    Filters out entire point_ids whose velocities deviate too strongly
    from the mean of other velocities within the same timepoint.

    Args:
        df: DataFrame with columns [point_id, cell_id, time, velocity]
        cell_ids: List of cell_ids to process
        filter_threshold: Discard point_id if any timepoint deviates by more than
                          n times the mean absolute deviation of others.
    Returns:
        Filtered DataFrame
    """
    df_filtered = df[df['cell_id'].isin(cell_ids)].copy()
    initial_count = df_filtered['point_id'].nunique()

    for cell_id in cell_ids:
        cell_mask = df_filtered['cell_id'] == cell_id
        cell_data = df_filtered[cell_mask].copy()

        iteration = 0
        while True:
            iteration += 1
            worst_point_id = None
            worst_deviation_ratio = 0

            for time_point in cell_data['time'].unique():
                time_data = cell_data[cell_data['time'] == time_point].dropna(subset=['velocity'])

                if len(time_data) <= 1:
                    continue

                for idx, point_id in zip(time_data.index, time_data['point_id']):
                    other_points = time_data[time_data.index != idx]
                    mean_velocity = other_points['velocity'].mean()
                    point_velocity = time_data.loc[idx, 'velocity']

                    velocity_diff = abs(point_velocity - mean_velocity)
                    mean_diff = abs(other_points['velocity'] - mean_velocity).mean()

                    if mean_diff > 0:
                        deviation_ratio = velocity_diff / mean_diff
                        if deviation_ratio > filter_threshold and deviation_ratio > worst_deviation_ratio:
                            worst_deviation_ratio = deviation_ratio
                            worst_point_id = point_id

            if worst_point_id is None:
                break

            print(f"  Iteration {iteration}: removed point {worst_point_id} "
                  f"(deviation ratio: {worst_deviation_ratio:.2f}) from {cell_id}")
            cell_data = cell_data[cell_data['point_id'] != worst_point_id]
            df_filtered = df_filtered[~((df_filtered['cell_id'] == cell_id) & 
                                       (df_filtered['point_id'] == worst_point_id))]

    final_count = df_filtered['point_id'].nunique()
    print(f"Total filtered out: {initial_count - final_count} "
          f"point(s) based on velocity threshold {filter_threshold}")
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Filter outlier trajectories based on velocity deviation")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file for filtered data")
    parser.add_argument("-c", "--cell-ids", nargs='+', help="Cell IDs to process (default: all cells in data)")
    parser.add_argument("-f", "--filter", type=float, required=True, help="Velocity deviation threshold")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.cell_ids:
        cell_ids = args.cell_ids
    else:
        cell_ids = df['cell_id'].unique()
        print(f"No cell_ids specified, filtering all {len(cell_ids)} cells")

    df_filtered = filter_velocity_outliers(df, cell_ids, args.filter)

    # Filter invalid positions
    df_filtered = df_filtered[df_filtered['x_parallel'] >= 0]

    df_filtered.to_csv(args.output, index=False)
    print(f"Filtered data saved to {args.output}")


if __name__ == "__main__":
    main()