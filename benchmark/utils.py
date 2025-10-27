import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_points_vs_x_parallel(df: pd.DataFrame, bins: int = None, output_path: str = None, cell_ids: list = None) -> None:
    """
    Plots the number of unique points by their x_parallel position.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_parallel', 'cell_id', 'frame', and 'point_id' columns.
    bins (int): Number of bins for x_parallel.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_parallel'])

    # Only look at the first timepoint in each cell (right before cut)
    first_timepoints = df.groupby('cell_id')['frame'].min().reset_index()
    df = df.merge(first_timepoints, on=['cell_id', 'frame'])

    plt.figure(figsize=(9.5, 5), dpi=300)
    plt.hist(df['x_parallel'], bins=bins, edgecolor='black')
    plt.xlabel('x_parallel (fraction of cut length)')
    plt.ylabel('Number of points')
    plt.title('Number of points by x_parallel', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return


def plot_points_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, bins: int = None, output_path: str = None, cell_ids: list = None) -> None:
    """
    Plots the number of unique points by their x_perpendicular position.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'cell_id', 'frame', and 'point_id' columns.
    bins (int): Number of bins for x_perpendicular.
    abs (bool): If True, plot using absolute values of x_perpendicular.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_perpendicular'])

    # Only look at the first timepoint in each cell (right before cut)
    first_timepoints = df.groupby('cell_id')['frame'].min().reset_index()
    df = df.merge(first_timepoints, on=['cell_id', 'frame'])

    plt.figure(figsize=(9.5, 5), dpi=300)

    if abs:
        df['x_perpendicular'] = df['x_perpendicular'].abs()

    plt.title('Number of points by |x_perpendicular|' if abs else 'Number of points by x_perpendicular', fontweight='bold')
    plt.xlabel('|x_perpendicular|' if abs else 'x_perpendicular')

    plt.hist(df['x_perpendicular'], bins=bins, edgecolor='black')
    plt.ylabel('Number of points')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return


def animate_points_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, bin_length: float = 5.0, 
                                       start_offset: int = -1, end_offset: int = None,
                                       output_path: str = None, cell_ids: list = None, fps: int = 5):
    """
    Creates an animation showing the distribution of points by x_perpendicular over time offsets.

    Uses the existing 'frame_rel' column (relative frame to cut) instead of computing offsets from 'cut'.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'cell_id', 'frame_rel', 'cut', and 'point_id' columns.
    abs (bool): If True, plot using absolute values of x_perpendicular. Default is True.
    bin_length (float): Width of histogram bins in pixels. Default is 5.0.
    start_offset (int): Starting time offset relative to cut. Default is -1.
    end_offset (int): Ending time offset relative to cut. If None, uses max offset in data.
    output_path (str): Path to save the animation as GIF. If None, returns HTML object for display.
    cell_ids (list): List of cell IDs to include. If None, all cells are included.
    fps (int): Frames per second for the animation. Default is 5.

    Returns:
    HTML or None: IPython HTML object containing the animation (or None if saved to file)
    """
    from IPython.display import HTML
    from matplotlib.animation import PillowWriter
    import matplotlib.animation as animation

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns (use frame_rel already present)
    df = df.dropna(subset=['x_perpendicular', 'frame_rel'])

    # Exclude actual cut frames
    if 'cut' in df.columns:
        df = df[df['cut'] == False]

    # Determine offset range using frame_rel directly
    if end_offset is None:
        end_offset = int(df['frame_rel'].max())

    offsets = np.arange(start_offset, end_offset + 1)

    # Set up bins based on abs parameter
    if abs:
        max_val = np.ceil(df['x_perpendicular'].abs().max()) if not df['x_perpendicular'].abs().empty else 0.0
        bins = np.arange(0.0, max_val + bin_length, bin_length)
    else:
        min_val = np.floor(df['x_perpendicular'].min()) if not df['x_perpendicular'].empty else 0.0
        max_val = np.ceil(df['x_perpendicular'].max()) if not df['x_perpendicular'].empty else 0.0
        bins = np.arange(min_val, max_val + bin_length, bin_length)

    # Compute max y for fixed axis
    max_y = 0
    for offset in offsets:
        tdf = df[df['frame_rel'] == offset]
        data = tdf['x_perpendicular'].abs() if abs else tdf['x_perpendicular']
        counts, _ = np.histogram(data, bins=bins)
        if len(counts) > 0 and counts.max() > max_y:
            max_y = counts.max()

    max_y = max_y * 1.05  # add 5% margin
    max_y = max(1, max_y)  # avoid zero

    # Create figure and animation
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)

    def update(i):
        offset = offsets[i]
        tdf = df[df['frame_rel'] == offset]
        ax.cla()

        data = tdf['x_perpendicular'].abs() if abs else tdf['x_perpendicular']
        ax.hist(data, bins=bins, edgecolor='black')

        xlabel = '|x_perpendicular| (pixels)' if abs else 'x_perpendicular (pixels)'
        title = f'Number of points by |x_perpendicular| at frame_rel {offset}' if abs else f'Number of points by x_perpendicular at frame_rel {offset}'

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of points')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.text(0.99, 0.95, f'frame_rel={offset} total points={tdf["point_id"].nunique()}',
                transform=ax.transAxes, ha='right', va='top')

        if abs:
            ax.set_xlim(-bin_length, bins.max())
        else:
            ax.set_xlim(bins.min() - bin_length/2, bins.max() + bin_length/2)

        ax.set_ylim(0, max_y + 1)

        # Add vertical red dashed lines for medians
        if not abs:
            less = tdf[tdf['x_perpendicular'] < 0]['x_perpendicular']
            more = tdf[tdf['x_perpendicular'] > 0]['x_perpendicular']
            if not less.empty:
                median_less = np.median(less)
                ax.axvline(median_less, color='red', linestyle='--', linewidth=2, label='Median')
            if not more.empty:
                median_more = np.median(more)
                ax.axvline(median_more, color='red', linestyle='--', linewidth=2,)
        else:
            # For abs, just one median
            if not data.empty:
                median_abs = np.median(data)
                ax.axvline(median_abs, color='red', linestyle='--', linewidth=2, label='Median')

        # Only show legend on first frame if any median lines were drawn
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        return ax.patches

    anim = animation.FuncAnimation(fig, update, frames=len(offsets), repeat=False)

    # Save or return
    if output_path:
        # Determine file extension
        ext = output_path.lower().split('.')[-1]
        if ext == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        else:
            # Default to mp4 if not gif
            if not output_path.lower().endswith('.mp4'):
                output_path = output_path + '.mp4'
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output_path, writer=writer)
        plt.close(fig)
        return None
    else:
        html = HTML(anim.to_jshtml(fps=fps))
        plt.close(fig)
        return html


def plot_points_vs_frame(df: pd.DataFrame, cell_ids: list = None, output_path: str = None) -> None:
    """
    Plots the number of unique points over frames.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'cell_id', and 'point_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel'])

    plt.figure(figsize=(9.5, 5), dpi=300)
    bins = np.arange(df['frame_rel'].min() - 0.5, df['frame_rel'].max() + 1.5, 1)
    plt.hist(df['frame_rel'], bins=bins, edgecolor='black')
    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Number of unique points')
    plt.title('Number of unique points over time', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(np.arange(df['frame_rel'].min(), df['frame_rel'].max() + 1, 1))

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return


def plot_velocities_vs_x_perpendicular(df: pd.DataFrame, abs: bool = True, frame_rel: int = None, cell_ids: list = None, output_path: str = None) -> None:
    """
    Plots the velocities of points by their x_perpendicular position at a specific frame_rel.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'x_perpendicular', 'velocity', and 'frame_rel' columns.
    abs (bool): If True, plot using absolute values of x_perpendicular.
    frame_rel (int): The frame_rel to filter the data. If None, uses all frames.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """
    
    df = df.copy()

    # Filter by frame_rel
    if frame_rel is not None:
        df = df[df['frame_rel'] == frame_rel]

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['x_perpendicular', 'velocity_cut'])

    plt.figure(figsize=(9.5, 5), dpi=300)

    if abs:
        df['x_perpendicular'] = df['x_perpendicular'].abs()

    plt.scatter(df['x_perpendicular'], df['velocity_cut'], alpha=0.6)
    plt.xlabel('|x_perpendicular|' if abs else 'x_perpendicular')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title(f'Point Velocities by {"|x_perpendicular|" if abs else "x_perpendicular"} at frame_rel {frame_rel}', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return


def plot_velocity_cut_vs_time(df: pd.DataFrame, cell_ids: list = None, output_path: str = None) -> None:
    """
    Plots the velocities of points over time (frame_rel) as lines per point, and overlays the average velocity.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'velocity_cut', and 'cell_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.

    Returns:
    None
    """

    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel', 'velocity_cut', 'point_id'])

    plt.figure(figsize=(9.5, 5), dpi=300)

    # Plot each point's velocity as a line
    for pid, group in df.groupby('point_id'):
        plt.plot(group['frame_rel'], group['velocity_cut'], color='gray', alpha=0.05, linewidth=0.5)

    # Plot the average velocity as a bold line
    avg = df.groupby('frame_rel')['velocity_cut'].mean()
    plt.plot(avg.index, avg.values, color='red', linewidth=2.5, label='Average velocity')

    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Velocity (pixels/frame)')
    plt.title('Point Velocities (relative to cut) over Time', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return


import matplotlib.pyplot as plt
import pandas as pd

def plot_v_perpendicular_vs_time(
    df: pd.DataFrame,
    cell_ids: list = None,
    output_path: str = None,
    abs: bool = True
) -> None:
    """
    Plots the perpendicular velocities of points over time (frame_rel) as lines per point,
    and overlays the average perpendicular velocity.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'frame_rel', 'v_perpendicular', and 'cell_id' columns.
    cell_ids (list): List of cell IDs to include in the plot. If None, all cells are included.
    output_path (str): Path to save the plot. If None, the plot is shown instead.
    abs (bool): If True, compute the average of the absolute perpendicular velocities.

    Returns:
    None
    """
    df = df.copy()

    # Filter by cell_ids if provided
    if cell_ids is not None:
        df = df[df['cell_id'].isin(cell_ids)]

    # Drop NaNs in required columns
    df = df.dropna(subset=['frame_rel', 'v_perpendicular', 'point_id'])

    if abs:
        df['v_perpendicular'] = df['v_perpendicular'].abs()

    plt.figure(figsize=(9.5, 5), dpi=300)

    # Plot each point's perpendicular velocity as a faint line
    for pid, group in df.groupby('point_id'):
        plt.plot(group['frame_rel'], group['v_perpendicular'], color='gray', alpha=0.05, linewidth=0.5)

    # Compute average across frames
    avg = df.groupby('frame_rel')['v_perpendicular'].mean()

    # Label depends on abs setting
    label = 'Average |v_perpendicular|' if abs else 'Average v_perpendicular'

    # Plot the average velocity
    plt.plot(avg.index, avg.values, color='blue', linewidth=2.5, label=label)

    plt.xlabel('Frame (relative to cut)')
    plt.ylabel('Perpendicular Velocity (pixels/frame)')
    plt.title('Point Perpendicular Velocities over Time', fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()

    return
