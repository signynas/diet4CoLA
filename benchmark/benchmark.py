import utils

if __name__ == "__main__":

    import os
    import sys
    import pandas as pd

    # Ensure output directory exists
    output_path = os.path.join("benchmark", "plots")
    os.makedirs(output_path, exist_ok=True)

    # Require path to points.csv as argument
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <path/to/points.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Check for required columns before plotting
    required_columns = [
        'x_parallel', 'x_perpendicular', 'cut', 'frame_rel', 'velocity_cut', 'point_id', 'cell_id', 'frame'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: The following required columns are missing from the input CSV: {', '.join(missing)}")
        sys.exit(1)

    # Plot 1: Number of points by x_parallel
    utils.plot_points_by_x_parallel(df, bins=12, output_path=os.path.join(output_path, "points_by_x_parallel.png"))

    # Plot 2: Number of points by x_perpendicular (abs=False)
    utils.plot_points_by_x_perpendicular(df, abs=False, bins=40, output_path=os.path.join(output_path, "points_by_x_perpendicular.png"))

    # Plot 3: Number of points by |x_perpendicular| (abs=True)
    utils.plot_points_by_x_perpendicular(df, abs=True, bins=20, output_path=os.path.join(output_path, "points_by_abs_x_perpendicular.png"))

    # Plot 4: Number of points by frame (excluding cut points)
    utils.plot_points_by_frame(df[df['cut'] == False], output_path=os.path.join(output_path, "points_by_frame.png"))

    # Plot 5: Velocities by x_perpendicular (abs=False, frame_rel=-1)
    utils.plot_velocities_by_x_perpendicular(df, abs=False, frame_rel=-1, output_path=os.path.join(output_path, "velocities_by_x_perpendicular_frame_rel_-1.png"))

    # Plot 6: Velocities by |x_perpendicular| (abs=True, frame_rel=-1)
    utils.plot_velocities_by_x_perpendicular(df, abs=True, frame_rel=-1, output_path=os.path.join(output_path, "velocities_by_abs_x_perpendicular_frame_rel_-1.png"))

    # Plot 7: Velocities over time
    utils.plot_velocities_over_time(df, output_path=os.path.join(output_path, "velocities_over_time.png"))

    # Animations
    utils.animate_points_by_x_perpendicular(df, abs=False, fps=3, output_path=os.path.join(output_path, "animation_points_by_x_perpendicular.mp4"))
    utils.animate_points_by_x_perpendicular(df, abs=True, fps=3, output_path=os.path.join(output_path, "animation_points_by_abs_x_perpendicular.mp4"))
