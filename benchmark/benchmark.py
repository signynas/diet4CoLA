import utils

if __name__ == "__main__":

    import os
    import sys
    import pandas as pd

    # Require path to points.csv and output directory as arguments
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <path/to/points.csv> <output_dir>")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected data file not found: {data_path}")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(data_path)

    # Check for required columns before plotting
    required_columns = [
        'x_parallel', 'x_perpendicular', 'cut', 'frame_rel', 'v_perpendicular', 'velocity_cut', 'point_id', 'cell_id', 'frame'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: The following required columns are missing from the input CSV: {', '.join(missing)}")
        sys.exit(1)

    # Plot 1: Number of points by x_parallel
    utils.plot_points_vs_x_parallel(df, bins=12, output_path=os.path.join(output_path, "points_by_x_parallel.png"))

    # Plot 2: Number of points by x_perpendicular (abs=False)
    utils.plot_points_vs_x_perpendicular(df, abs=False, bins=40, output_path=os.path.join(output_path, "points_by_x_perpendicular.png"))

    # Plot 3: Number of points by |x_perpendicular| (abs=True)
    utils.plot_points_vs_x_perpendicular(df, abs=True, bins=20, output_path=os.path.join(output_path, "points_by_abs_x_perpendicular.png"))

    # Plot 4: Number of points by frame (excluding cut points)
    utils.plot_points_vs_frame(df[df['cut'] == False], output_path=os.path.join(output_path, "points_by_frame.png"))

    # Plot 5: Velocities by x_perpendicular (abs=False, frame_rel=-1)
    utils.plot_velocities_vs_x_perpendicular(df, abs=False, frame_rel=-1, output_path=os.path.join(output_path, "velocities_by_x_perpendicular_frame_rel_-1.png"))

    # Plot 6: Velocities by |x_perpendicular| (abs=True, frame_rel=-1)
    utils.plot_velocities_vs_x_perpendicular(df, abs=True, frame_rel=-1, output_path=os.path.join(output_path, "velocities_by_abs_x_perpendicular_frame_rel_-1.png"))

    # Plot 7: Velocities relative to cut over time
    utils.plot_velocity_cut_vs_time(df, output_path=os.path.join(output_path, "velocities_over_time.png"))

    # Plot 8: Perpendicular velocities over time, absolute value
    utils.plot_v_perpendicular_vs_time(df, output_path=os.path.join(output_path, "v_perpendicular_over_time_abs.png"))
    
    # Plot 9: Perpendicular velocities over time, signed value
    utils.plot_v_perpendicular_vs_time(df, abs=False, output_path=os.path.join(output_path, "v_perpendicular_over_time_signed.png"))

    # Animations
    utils.animate_points_vs_x_perpendicular(df, abs=False, fps=3, output_path=os.path.join(output_path, "animation_points_by_x_perpendicular.mp4"))
    utils.animate_points_vs_x_perpendicular(df, abs=True, fps=3, output_path=os.path.join(output_path, "animation_points_by_abs_x_perpendicular.mp4"))
